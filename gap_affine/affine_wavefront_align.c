/*
 *                             The MIT License
 *
 * Wavefront Alignments Algorithms (VECT)
 * Copyright (c) 2024 by Yifei Guo  <guoyifei18@nudt.edu.cn>
 *
 * This file is part of Wavefront Alignments Algorithms (VECT).
 * The code is modified from WFA-paper(https://github.com/smarco/WFA-paper.) 
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * PROJECT: Wavefront Alignments Algorithms (VECT)
 * AUTHOR(S): Yifei Guo  <guoyifei18@nudt.edu.cn>
 * DESCRIPTION: Vectorized package
 */

#include "affine_wavefront_align.h"
#include "gap_affine/affine_wavefront_backtrace.h"
#include "gap_affine/affine_wavefront_display.h"
#include "gap_affine/affine_wavefront_extend.h"
#include "gap_affine/affine_wavefront_utils.h"
#include "utils/string_padded.h"
#include <immintrin.h>

#ifdef ENABLE_AVX512
#define SIMD_REG_SIZE 512
#define NUM_FOR_32 16
typedef __m512i __mxxxi;
#elif ENABLE_AVX2
#define SIMD_REG_SIZE 256
#define NUM_FOR_32 8
typedef __m256i __mxxxi;
#endif

#ifdef ENABLE_AVX512
typedef __mmask16 mmask_t;
static inline __mxxxi vect_add(const __mxxxi &a, const __mxxxi &b) { return _mm512_add_epi32(a, b); }
static inline __mxxxi vect_set(int32_t a15, int32_t a14, int32_t a13, int32_t a12, int32_t a11, int32_t a10, int32_t a9, int32_t a8, int32_t a7, int32_t a6, int32_t a5, int32_t a4, int32_t a3, int32_t a2, int32_t a1, int32_t a0) { return _mm256_set_epi32(a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0); }
static inline __mxxxi vect_and(const __mxxxi &a, const __mxxxi &b) { return _mm512_and_si512(a, b); }
static inline __mxxxi vect_set1(int32_t a) { return _mm512_set1_epi32(a); }
static inline __mxxxi vect_max(const __mxxxi &a, const __mxxxi &b) { return _mm512_max_epi32(a, b); }
static inline void vect_store(__mxxxi *mem_addr, const __mxxxi &a) { _mm512_storeu_si512(mem_addr, a); }
static inline __mxxxi vect_load(const __mxxxi *mem_addr) { return _mm512_loadu_si512(mem_addr); }
static inline __mxxxi vect_cmpgt(const __mxxxi &a, const __mxxxi &b) { return _mm512_cmpgt_epi32(a, b); }
static inline __mxxxi vect_cmpeq(const __mxxxi &a, const __mxxxi &b) { return _mm512_cmpeq_epi32(a, b); }
static inline __mxxxi vect_blend(mmask_t k, const __mxxxi &a, const __mxxxi &b) { return _mm512_mask_blend_epi32(k, a, b); }
#elif ENABLE_AVX2
static inline __mxxxi vect_add(const __mxxxi a, const __mxxxi b) { return _mm256_add_epi32(a, b); }
static inline __mxxxi vect_set(int32_t a7, int32_t a6, int32_t a5, int32_t a4, int32_t a3, int32_t a2, int32_t a1, int32_t a0) { return _mm256_set_epi32(a7, a6, a5, a4, a3, a2, a1, a0); }
static inline __mxxxi vect_and(const __mxxxi a, const __mxxxi b) { return _mm256_and_si256(a, b); }
static inline __mxxxi vect_set1(int32_t a) { return _mm256_set1_epi32(a); }
static inline __mxxxi vect_max(const __mxxxi a, const __mxxxi b) { return _mm256_max_epi32(a, b); }
static inline void vect_store(__mxxxi *mem_addr, const __mxxxi a) { _mm256_storeu_si256(mem_addr, a); }
static inline __mxxxi vect_load(const __mxxxi *mem_addr) { return _mm256_loadu_si256(mem_addr); }
static inline __mxxxi vect_cmpgt(const __mxxxi a, const __mxxxi b) { return _mm256_cmpgt_epi32(a, b); }
static inline __mxxxi vect_cmpeq(const __mxxxi a, const __mxxxi b) { return _mm256_cmpeq_epi32(a, b); }
static inline __mxxxi vect_blend(const __mxxxi k, const __mxxxi a, const __mxxxi b) { return _mm256_blendv_epi8(a, b, k); }
#endif

/*
 * Fetch & allocate wavefronts
 */
void affine_wavefronts_fetch_wavefronts(
    affine_wavefronts_t *const affine_wavefronts,
    affine_wavefront_set *const wavefront_set,
    const int score)
{
    // Compute scores
    const affine_penalties_t *const wavefront_penalties = &(affine_wavefronts->penalties.wavefront_penalties);
    const int mismatch_score = score - wavefront_penalties->mismatch;
    const int gap_open_score = score - wavefront_penalties->gap_opening - wavefront_penalties->gap_extension;
    const int gap_extend_score = score - wavefront_penalties->gap_extension;
    // Fetch wavefronts
    wavefront_set->in_mwavefront_sub = affine_wavefronts_get_source_mwavefront(affine_wavefronts, mismatch_score);
    wavefront_set->in_mwavefront_gap = affine_wavefronts_get_source_mwavefront(affine_wavefronts, gap_open_score);
    wavefront_set->in_iwavefront_ext = affine_wavefronts_get_source_iwavefront(affine_wavefronts, gap_extend_score);
    wavefront_set->in_dwavefront_ext = affine_wavefronts_get_source_dwavefront(affine_wavefronts, gap_extend_score);
}
void affine_wavefronts_allocate_wavefronts(
    affine_wavefronts_t *const affine_wavefronts,
    affine_wavefront_set *const wavefront_set,
    const int score,
    const int lo_effective,
    const int hi_effective)
{
    // Allocate M-Wavefront
    wavefront_set->out_mwavefront =
        affine_wavefronts_allocate_wavefront(affine_wavefronts, lo_effective, hi_effective);
    affine_wavefronts->mwavefronts[score] = wavefront_set->out_mwavefront;
    // Allocate I-Wavefront
    if (!wavefront_set->in_mwavefront_gap->null || !wavefront_set->in_iwavefront_ext->null)
    {
        wavefront_set->out_iwavefront =
            affine_wavefronts_allocate_wavefront(affine_wavefronts, lo_effective, hi_effective);
        affine_wavefronts->iwavefronts[score] = wavefront_set->out_iwavefront;
    }
    else
    {
        wavefront_set->out_iwavefront = NULL;
    }
    // Allocate D-Wavefront
    if (!wavefront_set->in_mwavefront_gap->null || !wavefront_set->in_dwavefront_ext->null)
    {
        wavefront_set->out_dwavefront =
            affine_wavefronts_allocate_wavefront(affine_wavefronts, lo_effective, hi_effective);
        affine_wavefronts->dwavefronts[score] = wavefront_set->out_dwavefront;
    }
    else
    {
        wavefront_set->out_dwavefront = NULL;
    }
}
void affine_wavefronts_compute_limits(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int score,
    int *const lo_effective,
    int *const hi_effective)
{
    // Set limits (min_lo)
    int lo = wavefront_set->in_mwavefront_sub->lo;
    if (lo > wavefront_set->in_mwavefront_gap->lo)
        lo = wavefront_set->in_mwavefront_gap->lo;
    if (lo > wavefront_set->in_iwavefront_ext->lo)
        lo = wavefront_set->in_iwavefront_ext->lo;
    if (lo > wavefront_set->in_dwavefront_ext->lo)
        lo = wavefront_set->in_dwavefront_ext->lo;
    --lo;
    // Set limits (max_hi)
    int hi = wavefront_set->in_mwavefront_sub->hi;
    if (hi < wavefront_set->in_mwavefront_gap->hi)
        hi = wavefront_set->in_mwavefront_gap->hi;
    if (hi < wavefront_set->in_iwavefront_ext->hi)
        hi = wavefront_set->in_iwavefront_ext->hi;
    if (hi < wavefront_set->in_dwavefront_ext->hi)
        hi = wavefront_set->in_dwavefront_ext->hi;
    ++hi;
    // Set effective limits values
    *hi_effective = hi;
    *lo_effective = lo;
}

/*
 * Compute wavefront offsets
 */
#define AFFINE_WAVEFRONT_DECLARE(wavefront, prefix)                  \
    const awf_offset_t *const prefix##_offsets = wavefront->offsets; \
    const int prefix##_hi = wavefront->hi;                           \
    const int prefix##_lo = wavefront->lo
#define AFFINE_WAVEFRONT_COND_FETCH(prefix, index, value) \
    (prefix##_lo <= (index) && (index) <= prefix##_hi) ? (value) : AFFINE_WAVEFRONT_OFFSET_NULL

#ifdef ENABLE_AVX2
/***
 * 拓展 WFA，计算 M 矩阵的部分，使用 AVX2 向量化实现
 * 对于核心计算部分 (m_sub_lo <= (k) && (k) <= m_sub_hi) ? (m_sub_offsets[k] + 1) : ((-2147483647-1)/2)
 * 拆分为 if bool then a else b，计算 bool = (m_sub_lo <= (k) && (k) <= m_sub_hi)
 ***/
void affine_wavefronts_compute_offsets_m(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo, const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

    // Compute score wavefronts
    int k;
    // AVX2每次处理的元素数目，256/32=8
    const int batchSize = NUM_FOR_32;
    // 设置向量，8个 sub_lo、sub_hi、1、-1、-MIN/2
    const __mxxxi loVec = vect_set1(m_sub_lo);
    const __mxxxi hiVec = vect_set1(m_sub_hi);
    const __mxxxi offsetAddition1 = vect_set1(1);
    const __mxxxi offsetMIN = vect_set1(-1073741824);
    __mxxxi k_vec;

    for (k = lo; k <= hi - batchSize + 1; k += batchSize)
    {
        // 设置向量单元：，m_sub_offsets[k:k+7]
        __mxxxi offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, offsetAddition1);

        // 用于后续的 mask,当 k在[m_sub_lo, m_sub_hi]之中。
        // 正向掩码若是设为全 1，否则设为 0；反向掩码若否设为 1，是设为 0
        k_vec = vect_set(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        __mxxxi mask = vect_and(vect_cmpgt(hiVec, k_vec), vect_cmpgt(k_vec, loVec));

        // _mm256_blendv_epi8(a,b,mask)  if mask then b else a（掩码为 1则为 offsets_vec）
        __mxxxi result = vect_blend(offsetMIN, offsets_vec, mask);

        // 存回数据
        vect_store((__mxxxi *)&out_moffsets[k], result);
    }

    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        out_moffsets[k] = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
    }
}

void affine_wavefronts_compute_offsets_dm(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo, const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_dwavefront_ext, d_ext);
    awf_offset_t *const out_doffsets = wavefront_set->out_dwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

    // 设置向量
    const __mxxxi m_sub_loVec = vect_set1(m_sub_lo);
    const __mxxxi m_sub_hiVec = vect_set1(m_sub_hi);
    const __mxxxi m_gap_loVec = vect_set1(m_gap_lo);
    const __mxxxi m_gap_hiVec = vect_set1(m_gap_hi);
    const __mxxxi d_ext_loVec = vect_set1(d_ext_lo);
    const __mxxxi d_ext_hiVec = vect_set1(d_ext_hi);
    const __mxxxi offsetMIN = vect_set1(-1073741824);
    __mxxxi offsets_vec, k_vec, mask, result;

    // Compute score wavefronts
    int k;
    // AVX2每次处理的元素数目，256/32=8
    const int batchSize = NUM_FOR_32;

    for (k = lo; k <= hi - batchSize + 1; k += batchSize)
    {

        // m_gap
        // m_gap_offsets[k+1:k+8]
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = vect_set(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_and(vect_cmpgt(m_gap_hiVec, k_vec), vect_cmpgt(k_vec, m_gap_loVec));
        // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi del_g = vect_blend(offsetMIN, offsets_vec, mask);

        // d_ext
        offsets_vec = vect_load((__mxxxi *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = vect_set(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_and(vect_cmpgt(d_ext_hiVec, k_vec), vect_cmpgt(k_vec, d_ext_loVec));
        // sub <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __mxxxi del_d = vect_blend(offsetMIN, offsets_vec, mask);

        // Update D
        __mxxxi del = vect_max(del_g, del_d);
        vect_store((__mxxxi *)&out_doffsets[k], del);

        // m_sub
        // m_sub_offsets[k:k+7]
        offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, vect_set1(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = vect_set(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_and(vect_cmpgt(m_sub_hiVec, k_vec), vect_cmpgt(k_vec, m_sub_loVec));
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __mxxxi sub = vect_blend(offsetMIN, offsets_vec, mask);

        // Update D
        __mxxxi max_m = vect_max(del, sub);
        vect_store((__mxxxi *)&out_moffsets[k], max_m);
    }

    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, sub);
    }
}

void affine_wavefronts_compute_offsets_im(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo, const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_iwavefront_ext, i_ext);
    awf_offset_t *const out_ioffsets = wavefront_set->out_iwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

    // 设置向量
    const __mxxxi m_sub_loVec = vect_set1(m_sub_lo);
    const __mxxxi m_sub_hiVec = vect_set1(m_sub_hi);
    const __mxxxi m_gap_loVec = vect_set1(m_gap_lo);
    const __mxxxi m_gap_hiVec = vect_set1(m_gap_hi);
    const __mxxxi i_ext_loVec = vect_set1(i_ext_lo);
    const __mxxxi i_ext_hiVec = vect_set1(i_ext_hi);
    const __mxxxi offsetMIN = vect_set1(-1073741824);
    __mxxxi offsets_vec, k_vec, mask, result;

    // Compute score wavefronts
    int k = lo;

    // 第一次计算
    // Update I
    const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
    const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
    const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
    out_ioffsets[k] = ins;
    // Update M
    const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
    out_moffsets[k] = MAX(ins, sub);
    k++;

    // AVX2每次处理的元素数目，256/32=8
    const int batchSize = NUM_FOR_32;
    // 向量化部分
    for (; k <= hi - batchSize + 1; k += batchSize)
    {

        // Update I
        // m_gap_offsets[k-1:k+6]
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k - 1]);
        // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
        k_vec = vect_set(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_and(vect_cmpgt(m_gap_hiVec, k_vec), vect_cmpgt(k_vec, m_gap_loVec));
        // sub <- if mask ? (m_gap_offsets[k-1]) : (-1073741824)
        __mxxxi ins_g = vect_blend(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = vect_load((__mxxxi *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = vect_set(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_and(vect_cmpgt(i_ext_hiVec, k_vec), vect_cmpgt(k_vec, i_ext_loVec));
        // sub <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __mxxxi ins_i = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi ins = vect_max(ins_g, ins_i);
        ins = vect_add(ins, vect_set1(1));
        vect_store((__mxxxi *)&out_ioffsets[k], ins);

        // m_sub
        // m_sub_offsets[k:k+7]
        offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, vect_set1(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = vect_set(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_and(vect_cmpgt(m_sub_hiVec, k_vec), vect_cmpgt(k_vec, m_sub_loVec));
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __mxxxi sub = vect_blend(offsetMIN, offsets_vec, mask);

        // Update D
        __mxxxi max_m = vect_max(ins, sub);
        vect_store((__mxxxi *)&out_moffsets[k], max_m);
    }

    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(ins, sub);
    }
}

void affine_wavefronts_compute_offsets_idm(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo,
    const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_iwavefront_ext, i_ext);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_dwavefront_ext, d_ext);
    awf_offset_t *const out_ioffsets = wavefront_set->out_iwavefront->offsets;
    awf_offset_t *const out_doffsets = wavefront_set->out_dwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;
    // Compute loop peeling offset (min_hi)
    int min_hi = wavefront_set->in_mwavefront_sub->hi;
    if (!wavefront_set->in_mwavefront_gap->null && min_hi > wavefront_set->in_mwavefront_gap->hi - 1)
        min_hi = wavefront_set->in_mwavefront_gap->hi - 1;
    if (!wavefront_set->in_iwavefront_ext->null && min_hi > wavefront_set->in_iwavefront_ext->hi + 1)
        min_hi = wavefront_set->in_iwavefront_ext->hi + 1;
    if (!wavefront_set->in_dwavefront_ext->null && min_hi > wavefront_set->in_dwavefront_ext->hi - 1)
        min_hi = wavefront_set->in_dwavefront_ext->hi - 1;
    // Compute loop peeling offset (max_lo)
    int max_lo = wavefront_set->in_mwavefront_sub->lo;
    if (!wavefront_set->in_mwavefront_gap->null && max_lo < wavefront_set->in_mwavefront_gap->lo + 1)
        max_lo = wavefront_set->in_mwavefront_gap->lo + 1;
    if (!wavefront_set->in_iwavefront_ext->null && max_lo < wavefront_set->in_iwavefront_ext->lo + 1)
        max_lo = wavefront_set->in_iwavefront_ext->lo + 1;
    if (!wavefront_set->in_dwavefront_ext->null && max_lo < wavefront_set->in_dwavefront_ext->lo - 1)
        max_lo = wavefront_set->in_dwavefront_ext->lo - 1;
    // Compute score wavefronts (prologue)

    // 设置向量
    const __mxxxi m_sub_loVec = vect_set1(m_sub_lo);
    const __mxxxi m_sub_hiVec = vect_set1(m_sub_hi);
    const __mxxxi m_gap_loVec = vect_set1(m_gap_lo);
    const __mxxxi m_gap_hiVec = vect_set1(m_gap_hi);
    const __mxxxi i_ext_loVec = vect_set1(i_ext_lo);
    const __mxxxi i_ext_hiVec = vect_set1(i_ext_hi);
    const __mxxxi d_ext_loVec = vect_set1(d_ext_lo);
    const __mxxxi d_ext_hiVec = vect_set1(d_ext_hi);
    const __mxxxi offsetMIN = vect_set1(-1073741824);
    __mxxxi offsets_vec, k_vec, mask;
    // AVX2每次处理的元素数目，256/32=8
    const int batchSize = NUM_FOR_32;

    int k = lo;
    // 第一次计算
    // Compute score wavefronts
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, MAX(sub, ins));
        k++;
    }
    // 向量化部分
    for (; k < max_lo - batchSize + 1; k += batchSize)
    {
        // Update I
        // ins_g
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k - 1]);
        // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
        k_vec = vect_set(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_and(vect_cmpgt(m_gap_hiVec, k_vec), vect_cmpgt(k_vec, m_gap_loVec));
        // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi ins_g = vect_blend(offsetMIN, offsets_vec, mask);
        // ins_i
        offsets_vec = vect_load((__mxxxi *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = vect_set(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_and(vect_cmpgt(i_ext_hiVec, k_vec), vect_cmpgt(k_vec, i_ext_loVec));
        // ins_i <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __mxxxi ins_i = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi ins = vect_max(ins_g, ins_i);
        ins = vect_add(ins, vect_set1(1));
        vect_store((__mxxxi *)&out_ioffsets[k], ins);

        // Update D
        // del_g
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = vect_set(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_and(vect_cmpgt(m_gap_hiVec, k_vec), vect_cmpgt(k_vec, m_gap_loVec));
        // del_g <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi del_g = vect_blend(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = vect_load((__mxxxi *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = vect_set(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_and(vect_cmpgt(d_ext_hiVec, k_vec), vect_cmpgt(k_vec, d_ext_loVec));
        // del_d <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __mxxxi del_d = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi del = vect_max(del_g, del_d);
        vect_store((__mxxxi *)&out_doffsets[k], del);

        // Update M
        // m_sub_offsets[k:k+7]
        offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, vect_set1(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = vect_set(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_and(vect_cmpgt(m_sub_hiVec, k_vec), vect_cmpgt(k_vec, m_sub_loVec));
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __mxxxi sub = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi max_m = vect_max(del, sub);
        max_m = vect_max(max_m, ins);
        vect_store((__mxxxi *)&out_moffsets[k], max_m);
    }
    // 计算剩余部分
    for (; k < max_lo; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }

    // Compute score wavefronts (core)

    // 向量化部分
    for (; k <= min_hi - batchSize + 1; k += batchSize)
    {
        // Update I
        __mxxxi m_gapi_value = vect_load((__mxxxi *)&m_gap_offsets[k - 1]);
        __mxxxi i_ext_value = vect_load((__mxxxi *)&i_ext_offsets[k - 1]);
        __mxxxi ins = vect_max(i_ext_value, m_gapi_value);
        ins = vect_add(ins, vect_set1(1));
        vect_store((__mxxxi *)&out_ioffsets[k], ins);
        // Update D
        __mxxxi m_gapd_value = vect_load((__mxxxi *)&m_gap_offsets[k + 1]);
        __mxxxi d_ext_value = vect_load((__mxxxi *)&d_ext_offsets[k + 1]);
        __mxxxi del = vect_max(m_gapd_value, d_ext_value);
        vect_store((__mxxxi *)&out_doffsets[k], del);
        // Update M
        __mxxxi sub = vect_load((__mxxxi *)&m_sub_offsets[k]);
        sub = vect_add(sub, vect_set1(1));
        sub = vect_max(sub, del);
        sub = vect_max(sub, ins);
        vect_store((__mxxxi *)&out_moffsets[k], sub);
    }
    for (; k <= min_hi; ++k)
    {
        // Update I
        const awf_offset_t m_gapi_value = m_gap_offsets[k - 1];
        const awf_offset_t i_ext_value = i_ext_offsets[k - 1];
        const awf_offset_t ins = MAX(m_gapi_value, i_ext_value) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t m_gapd_value = m_gap_offsets[k + 1];
        const awf_offset_t d_ext_value = d_ext_offsets[k + 1];
        const awf_offset_t del = MAX(m_gapd_value, d_ext_value);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = m_sub_offsets[k] + 1;
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }

    // Compute score wavefronts (epilogue)
    for (; k <= min_hi - batchSize + 1; k += batchSize)
    {
        // Update I
        // ins_g
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k - 1]);
        // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
        k_vec = vect_set(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_and(vect_cmpgt(m_gap_hiVec, k_vec), vect_cmpgt(k_vec, m_gap_loVec));
        // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi ins_g = vect_blend(offsetMIN, offsets_vec, mask);
        // ins_i
        offsets_vec = vect_load((__mxxxi *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = vect_set(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_and(vect_cmpgt(i_ext_hiVec, k_vec), vect_cmpgt(k_vec, i_ext_loVec));
        // ins_i <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __mxxxi ins_i = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi ins = vect_max(ins_g, ins_i);
        ins = vect_add(ins, vect_set1(1));
        vect_store((__mxxxi *)&out_ioffsets[k], ins);

        // Update D
        // del_g
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = vect_set(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_and(vect_cmpgt(m_gap_hiVec, k_vec), vect_cmpgt(k_vec, m_gap_loVec));
        // del_g <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi del_g = vect_blend(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = vect_load((__mxxxi *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = vect_set(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_and(vect_cmpgt(d_ext_hiVec, k_vec), vect_cmpgt(k_vec, d_ext_loVec));
        // del_d <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __mxxxi del_d = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi del = vect_max(del_g, del_d);
        vect_store((__mxxxi *)&out_doffsets[k], del);

        // Update M
        // m_sub_offsets[k:k+7]
        offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, vect_set1(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = vect_set(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_and(vect_cmpgt(m_sub_hiVec, k_vec), vect_cmpgt(k_vec, m_sub_loVec));
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __mxxxi sub = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi max_m = vect_max(del, sub);
        max_m = vect_max(max_m, ins);
        vect_store((__mxxxi *)&out_moffsets[k], max_m);
    }
    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }
}

#elif ENABLE_AVX512
void affine_wavefronts_compute_offsets_m(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo, const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

    // Compute score wavefronts
    int k;
    const int batchSize = NUM_FOR_32;
    // 设置向量，8个 sub_lo、sub_hi、1、-1、-MIN/2
    const __mxxxi loVec = vect_set1(m_sub_lo);
    const __mxxxi hiVec = vect_set1(m_sub_hi);
    const __mxxxi offsetAddition1 = vect_set1(1);
    const __mxxxi offsetMIN = vect_set1(-1073741824);
    __mxxxi k_vec;
    mmask_t mask;

    for (k = lo; k <= hi - batchSize + 1; k += batchSize)
    {
        // 设置向量单元：，m_sub_offsets[k:k+7]
        __mxxxi offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, offsetAddition1);

        // 用于后续的 mask,当 k在[m_sub_lo, m_sub_hi]之中。
        // 正向掩码若是设为全 1，否则设为 0；反向掩码若否设为 1，是设为 0
        k_vec = vect_set(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_cmpgt(hiVec, k_vec) & vect_cmpgt(k_vec, loVec);

        // _mm256_blendv_epi8(a,b,mask)  if mask then b else a（掩码为 1则为 offsets_vec）
        __mxxxi result = vect_blend(offsetMIN, offsets_vec, mask);

        // 存回数据
        vect_store((__mxxxi *)&out_moffsets[k], result);
    }

    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        out_moffsets[k] = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
    }
}

void affine_wavefronts_compute_offsets_dm(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo, const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_dwavefront_ext, d_ext);
    awf_offset_t *const out_doffsets = wavefront_set->out_dwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

    // 设置向量
    const __mxxxi m_sub_loVec = vect_set(m_sub_lo);
    const __mxxxi m_sub_hiVec = vect_set1(m_sub_hi);
    const __mxxxi m_gap_loVec = vect_set1(m_gap_lo);
    const __mxxxi m_gap_hiVec = vect_set1(m_gap_hi);
    const __mxxxi d_ext_loVec = vect_set1(d_ext_lo);
    const __mxxxi d_ext_hiVec = vect_set1(d_ext_hi);
    const __mxxxi offsetMIN = vect_set1(-1073741824);
    __mxxxi offsets_vec, k_vec, result;
    mmask_t mask;

    // Compute score wavefronts
    int k;
    // AVX2每次处理的元素数目，256/32=8
    const int batchSize = NUM_FOR_32;

    for (k = lo; k <= hi - batchSize + 1; k += batchSize)
    {

        // m_gap
        // m_gap_offsets[k+1:k+8]
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = vect_set(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_cmpgt(m_gap_hiVec, k_vec) & vect_cmpgt(k_vec, m_gap_loVec);
        // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi del_g = vect_blend(offsetMIN, offsets_vec, mask);

        // d_ext
        offsets_vec = vect_load((__mxxxi *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = vect_set(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_cmpgt(d_ext_hiVec, k_vec) & vect_cmpgt(k_vec, d_ext_loVec);
        // sub <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __mxxxi del_d = vect_blend(offsetMIN, offsets_vec, mask);

        // Update D
        __mxxxi del = vect_max(del_g, del_d);
        vect_store((__mxxxi *)&out_doffsets[k], del);

        // m_sub
        offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, vect_set1(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = vect_set(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_cmpgt(m_sub_hiVec, k_vec) & vect_cmpgt(k_vec, m_sub_loVec);
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __mxxxi sub = vect_blend(offsetMIN, offsets_vec, mask);

        // Update D
        __mxxxi max_m = vect_max(del, sub);
        vect_store((__mxxxi *)&out_moffsets[k], max_m);
    }

    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, sub);
    }
}

void affine_wavefronts_compute_offsets_im(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo, const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_iwavefront_ext, i_ext);
    awf_offset_t *const out_ioffsets = wavefront_set->out_iwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

    // 设置向量
    const __mxxxi m_sub_loVec = vect_set1(m_sub_lo);
    const __mxxxi m_sub_hiVec = vect_set1(m_sub_hi);
    const __mxxxi m_gap_loVec = vect_set1(m_gap_lo);
    const __mxxxi m_gap_hiVec = vect_set1(m_gap_hi);
    const __mxxxi i_ext_loVec = vect_set1(i_ext_lo);
    const __mxxxi i_ext_hiVec = vect_set1(i_ext_hi);
    const __mxxxi offsetMIN = vect_set1(-1073741824);
    __mxxxi offsets_vec, k_vec, result;
    mmask_t mask;

    // Compute score wavefronts
    int k = lo;

    // 第一次计算
    // Update I
    const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
    const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
    const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
    out_ioffsets[k] = ins;
    // Update M
    const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
    out_moffsets[k] = MAX(ins, sub);
    k++;

    // AVX2每次处理的元素数目，256/32=8
    const int batchSize = NUM_FOR_32;
    // 向量化部分
    for (; k <= hi - batchSize + 1; k += batchSize)
    {

        // Update I
        // m_gap_offsets[k-1:k+6]
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k - 1]);
        k_vec = vect_set(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
                         k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_cmpgt(m_gap_hiVec, k_vec) & vect_cmpgt(k_vec, m_gap_loVec);
        __mxxxi ins_g = vect_blend(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = vect_load((__mxxxi *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = vect_set(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
                         k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_cmpgt(i_ext_hiVec, k_vec) & vect_cmpgt(k_vec, i_ext_loVec);
        // sub <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __mxxxi ins_i = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi ins = vect_max(ins_g, ins_i);
        ins = vect_add(ins, vect_set(1));
        vect_store((__mxxxi *)&out_ioffsets[k], ins);

        // m_sub
        // m_sub_offsets[k:k+7]
        offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, vect_set(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = vect_set(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_cmpgt(m_sub_hiVec, k_vec) & vect_cmpgt(k_vec, m_sub_loVec);
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __mxxxi sub = vect_blend(offsetMIN, offsets_vec, mask);

        // Update D
        __mxxxi max_m = vect_max(ins, sub);
        vect_store((__mxxxi *)&out_moffsets[k], max_m);
    }

    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(ins, sub);
    }
}

void affine_wavefronts_compute_offsets_idm(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo,
    const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_iwavefront_ext, i_ext);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_dwavefront_ext, d_ext);
    awf_offset_t *const out_ioffsets = wavefront_set->out_iwavefront->offsets;
    awf_offset_t *const out_doffsets = wavefront_set->out_dwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;
    // Compute loop peeling offset (min_hi)
    int min_hi = wavefront_set->in_mwavefront_sub->hi;
    if (!wavefront_set->in_mwavefront_gap->null && min_hi > wavefront_set->in_mwavefront_gap->hi - 1)
        min_hi = wavefront_set->in_mwavefront_gap->hi - 1;
    if (!wavefront_set->in_iwavefront_ext->null && min_hi > wavefront_set->in_iwavefront_ext->hi + 1)
        min_hi = wavefront_set->in_iwavefront_ext->hi + 1;
    if (!wavefront_set->in_dwavefront_ext->null && min_hi > wavefront_set->in_dwavefront_ext->hi - 1)
        min_hi = wavefront_set->in_dwavefront_ext->hi - 1;
    // Compute loop peeling offset (max_lo)
    int max_lo = wavefront_set->in_mwavefront_sub->lo;
    if (!wavefront_set->in_mwavefront_gap->null && max_lo < wavefront_set->in_mwavefront_gap->lo + 1)
        max_lo = wavefront_set->in_mwavefront_gap->lo + 1;
    if (!wavefront_set->in_iwavefront_ext->null && max_lo < wavefront_set->in_iwavefront_ext->lo + 1)
        max_lo = wavefront_set->in_iwavefront_ext->lo + 1;
    if (!wavefront_set->in_dwavefront_ext->null && max_lo < wavefront_set->in_dwavefront_ext->lo - 1)
        max_lo = wavefront_set->in_dwavefront_ext->lo - 1;
    // Compute score wavefronts (prologue)

    // 设置向量
    const __mxxxi m_sub_loVec = vect_set1(m_sub_lo);
    const __mxxxi m_sub_hiVec = vect_set1(m_sub_hi);
    const __mxxxi m_gap_loVec = vect_set1(m_gap_lo);
    const __mxxxi m_gap_hiVec = vect_set1(m_gap_hi);
    const __mxxxi i_ext_loVec = vect_set1(i_ext_lo);
    const __mxxxi i_ext_hiVec = vect_set1(i_ext_hi);
    const __mxxxi d_ext_loVec = vect_set1(d_ext_lo);
    const __mxxxi d_ext_hiVec = vect_set1(d_ext_hi);
    const __mxxxi offsetMIN = vect_set1(-1073741824);
    __mxxxi offsets_vec, k_vec;

    mmask_t mask;
    const int batchSize = NUM_FOR_32;

    int k = lo;
    // 第一次计算
    // Compute score wavefronts
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, MAX(sub, ins));
        k++;
    }
    // 向量化部分
    for (; k < max_lo - batchSize + 1; k += batchSize)
    {
        // Update I
        // ins_g
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k - 1]);
        k_vec = vect_set(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
                         k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_cmpgt(m_gap_hiVec, k_vec) & vect_cmpgt(k_vec, m_gap_loVec);
        __mxxxi ins_g = vect_blend(offsetMIN, offsets_vec, mask);
        // ins_i
        offsets_vec = vect_load((__mxxxi *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = vect_set(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
                         k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_cmpgt(i_ext_hiVec, k_vec) & vect_cmpgt(k_vec, i_ext_loVec);
        // ins_i <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __mxxxi ins_i = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi ins = vect_max(ins_g, ins_i);
        ins = vect_add(ins, vect_set1(1));
        vect_store((__mxxxi *)&out_ioffsets[k], ins);

        // Update D
        // del_g
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = vect_set(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_cmpgt(m_gap_hiVec, k_vec) & vect_cmpgt(k_vec, m_gap_loVec);
        // del_g <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi del_g = vect_blend(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = vect_load((__mxxxi *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = vect_set(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_cmpgt(d_ext_hiVec, k_vec) & vect_cmpgt(k_vec, d_ext_loVec);
        // del_d <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __mxxxi del_d = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi del = vect_max(del_g, del_d);
        vect_store((__mxxxi *)&out_doffsets[k], del);

        // Update M
        // m_sub_offsets[k:k+7]
        offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, vect_set1(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = vect_set(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_cmpgt(m_sub_hiVec, k_vec) & vect_cmpgt(k_vec, m_sub_loVec);
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __mxxxi sub = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi max_m = vect_max(del, sub);
        max_m = vect_max(max_m, ins);
        vect_store((__mxxxi *)&out_moffsets[k], max_m);
    }
    // 计算剩余部分
    for (; k < max_lo; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }

    // Compute score wavefronts (core)

    // 向量化部分
    for (; k <= min_hi - batchSize + 1; k += batchSize)
    {
        // Update I
        __mxxxi m_gapi_value = vect_load((__mxxxi *)&m_gap_offsets[k - 1]);
        __mxxxi i_ext_value = vect_load((__mxxxi *)&i_ext_offsets[k - 1]);
        __mxxxi ins = vect_max(i_ext_value, m_gapi_value);
        ins = vect_add(ins, vect_set1(1));
        vect_store((__mxxxi *)&out_ioffsets[k], ins);
        // Update D
        __mxxxi m_gapd_value = vect_load((__mxxxi *)&m_gap_offsets[k + 1]);
        __mxxxi d_ext_value = vect_load((__mxxxi *)&d_ext_offsets[k + 1]);
        __mxxxi del = vect_max(m_gapd_value, d_ext_value);
        vect_store((__mxxxi *)&out_doffsets[k], del);
        // Update M
        __mxxxi sub = vect_load((__mxxxi *)&m_sub_offsets[k]);
        sub = vect_add(sub, vect_set1(1));
        sub = vect_max(sub, del);
        sub = vect_max(sub, ins);
        vect_store((__mxxxi *)&out_moffsets[k], sub);
    }
    for (; k <= min_hi; ++k)
    {
        // Update I
        const awf_offset_t m_gapi_value = m_gap_offsets[k - 1];
        const awf_offset_t i_ext_value = i_ext_offsets[k - 1];
        const awf_offset_t ins = MAX(m_gapi_value, i_ext_value) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t m_gapd_value = m_gap_offsets[k + 1];
        const awf_offset_t d_ext_value = d_ext_offsets[k + 1];
        const awf_offset_t del = MAX(m_gapd_value, d_ext_value);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = m_sub_offsets[k] + 1;
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }

    // Compute score wavefronts (epilogue)
    for (; k <= min_hi - batchSize + 1; k += batchSize)
    {
        // Update I
        // ins_g
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k - 1]);
        // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
        k_vec = vect_set(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
                         k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_cmpgt(m_gap_hiVec, k_vec) & vect_cmpgt(k_vec, m_gap_loVec);
        // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi ins_g = vect_blend(offsetMIN, offsets_vec, mask);
        // ins_i
        offsets_vec = vect_load((__mxxxi *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = vect_set(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
                         k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = vect_cmpgt(i_ext_hiVec, k_vec) & vect_cmpgt(k_vec, i_ext_loVec);
        // ins_i <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __mxxxi ins_i = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi ins = vect_max(ins_g, ins_i);
        ins = vect_add(ins, vect_set1(1));
        vect_store((__mxxxi *)&out_ioffsets[k], ins);

        // Update D
        // del_g
        offsets_vec = vect_load((__mxxxi *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = vect_set(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_cmpgt(m_gap_hiVec, k_vec) & vect_cmpgt(k_vec, m_gap_loVec);
        // del_g <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __mxxxi del_g = vect_blend(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = vect_load((__mxxxi *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = vect_set(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = vect_cmpgt(d_ext_hiVec, k_vec) & vect_cmpgt(k_vec, d_ext_loVec);
        // del_d <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __mxxxi del_d = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi del = vect_max(del_g, del_d);
        vect_store((__mxxxi *)&out_doffsets[k], del);

        // Update M
        // m_sub_offsets[k:k+7]
        offsets_vec = vect_load((__mxxxi *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = vect_add(offsets_vec, vect_set1(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = vect_set(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
                         k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = vect_cmpgt(m_sub_hiVec, k_vec) & vect_cmpgt(k_vec, m_sub_loVec);
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __mxxxi sub = vect_blend(offsetMIN, offsets_vec, mask);

        __mxxxi max_m = vect_max(del, sub);
        max_m = vect_max(max_m, ins);
        vect_store((__mxxxi *)&out_moffsets[k], max_m);
    }
    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }
}

#else
/*
 * Compute wavefront offsets
 */
void affine_wavefronts_compute_offsets_idm(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo,
    const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_iwavefront_ext, i_ext);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_dwavefront_ext, d_ext);
    awf_offset_t *const out_ioffsets = wavefront_set->out_iwavefront->offsets;
    awf_offset_t *const out_doffsets = wavefront_set->out_dwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;
    // Compute loop peeling offset (min_hi)
    int min_hi = wavefront_set->in_mwavefront_sub->hi;
    if (!wavefront_set->in_mwavefront_gap->null && min_hi > wavefront_set->in_mwavefront_gap->hi - 1)
        min_hi = wavefront_set->in_mwavefront_gap->hi - 1;
    if (!wavefront_set->in_iwavefront_ext->null && min_hi > wavefront_set->in_iwavefront_ext->hi + 1)
        min_hi = wavefront_set->in_iwavefront_ext->hi + 1;
    if (!wavefront_set->in_dwavefront_ext->null && min_hi > wavefront_set->in_dwavefront_ext->hi - 1)
        min_hi = wavefront_set->in_dwavefront_ext->hi - 1;
    // Compute loop peeling offset (max_lo)
    int max_lo = wavefront_set->in_mwavefront_sub->lo;
    if (!wavefront_set->in_mwavefront_gap->null && max_lo < wavefront_set->in_mwavefront_gap->lo + 1)
        max_lo = wavefront_set->in_mwavefront_gap->lo + 1;
    if (!wavefront_set->in_iwavefront_ext->null && max_lo < wavefront_set->in_iwavefront_ext->lo + 1)
        max_lo = wavefront_set->in_iwavefront_ext->lo + 1;
    if (!wavefront_set->in_dwavefront_ext->null && max_lo < wavefront_set->in_dwavefront_ext->lo - 1)
        max_lo = wavefront_set->in_dwavefront_ext->lo - 1;
    // Compute score wavefronts (prologue)
    int k;
    for (k = lo; k < max_lo; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }
    // Compute score wavefronts (core)
    for (k = max_lo; k <= min_hi; ++k)
    {
        // Update I
        const awf_offset_t m_gapi_value = m_gap_offsets[k - 1];
        const awf_offset_t i_ext_value = i_ext_offsets[k - 1];
        const awf_offset_t ins = MAX(m_gapi_value, i_ext_value) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t m_gapd_value = m_gap_offsets[k + 1];
        const awf_offset_t d_ext_value = d_ext_offsets[k + 1];
        const awf_offset_t del = MAX(m_gapd_value, d_ext_value);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = m_sub_offsets[k] + 1;
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }
    // Compute score wavefronts (epilogue)
    for (k = min_hi + 1; k <= hi; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, MAX(sub, ins));
    }
}

void affine_wavefronts_compute_offsets_im(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo,
    const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_iwavefront_ext, i_ext);
    awf_offset_t *const out_ioffsets = wavefront_set->out_iwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;
    // Compute score wavefronts
    int k;
    for (k = lo; k <= hi; ++k)
    {
        // Update I
        const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
        const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
        const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
        out_ioffsets[k] = ins;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(ins, sub);
    }
}

void affine_wavefronts_compute_offsets_dm(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo,
    const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_dwavefront_ext, d_ext);
    awf_offset_t *const out_doffsets = wavefront_set->out_dwavefront->offsets;
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;
    // Compute score wavefronts
    int k;
    for (k = lo; k <= hi; ++k)
    {
        // Update D
        const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
        const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
        const awf_offset_t del = MAX(del_g, del_d);
        out_doffsets[k] = del;
        // Update M
        const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
        out_moffsets[k] = MAX(del, sub);
    }
}

void affine_wavefronts_compute_offsets_m(
    affine_wavefronts_t *const affine_wavefronts,
    const affine_wavefront_set *const wavefront_set,
    const int lo,
    const int hi)
{
    // Parameters
    AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
    awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;
    // Compute score wavefronts
    int k;
    for (k = lo; k <= hi; ++k)
    {
        // Update M
        out_moffsets[k] = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
    }
}

#endif

/*
 * Compute wavefront
 */
void affine_wavefronts_compute_wavefront(
    affine_wavefronts_t *const affine_wavefronts,
    const char *const pattern,
    const int pattern_length,
    const char *const text,
    const int text_length,
    const int score)
{

    struct timeval start_alloc_time, end_alloc_time;
    struct timeval start_kernel_time, end_kernel_time;

    // Select wavefronts
    affine_wavefront_set wavefront_set;
    affine_wavefronts_fetch_wavefronts(affine_wavefronts, &wavefront_set, score);
    // Check null wavefronts
    if (wavefront_set.in_mwavefront_sub->null &&
        wavefront_set.in_mwavefront_gap->null &&
        wavefront_set.in_iwavefront_ext->null &&
        wavefront_set.in_dwavefront_ext->null)
    {
        WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_steps_null, 1);
        return;
    }
    WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_null_used, (wavefront_set.in_mwavefront_sub->null ? 1 : 0));
    WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_null_used, (wavefront_set.in_mwavefront_gap->null ? 1 : 0));
    WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_null_used, (wavefront_set.in_iwavefront_ext->null ? 1 : 0));
    WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_null_used, (wavefront_set.in_dwavefront_ext->null ? 1 : 0));
    // Set limits
    int hi, lo;
    affine_wavefronts_compute_limits(affine_wavefronts, &wavefront_set, score, &lo, &hi);
    // Allocate score-wavefronts

    affine_wavefronts_allocate_wavefronts(affine_wavefronts, &wavefront_set, score, lo, hi);

    // Compute WF
    const int kernel = ((wavefront_set.out_iwavefront != NULL) << 1) | (wavefront_set.out_dwavefront != NULL);
    WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_compute_kernel[kernel], 1);
    // #ifdef ENABLE_AVX512
    // // printf("AVX2");
    // switch (kernel)
    // {
    // case 3: // 11b
    //     affine_wavefronts_compute_offsets_idm_avx2(affine_wavefronts, &wavefront_set, lo, hi);
    //     break;
    // case 2: // 10b
    //     affine_wavefronts_compute_offsets_im_avx2(affine_wavefronts, &wavefront_set, lo, hi);
    //     break;
    // case 1: // 01b
    //     affine_wavefronts_compute_offsets_dm_avx2(affine_wavefronts, &wavefront_set, lo, hi);
    //     break;
    // case 0: // 00b
    //     affine_wavefronts_compute_offsets_m_avx2(affine_wavefronts, &wavefront_set, lo, hi);
    //     break;
    //     }
    // #elif ENABLE_AVX512
    //     // printf("AVX512");
    //     switch (kernel)
    //     {
    //     case 3: // 11b
    //         affine_wavefronts_compute_offsets_idm_avx512(affine_wavefronts, &wavefront_set, lo, hi);
    //         break;
    //     case 2: // 10b
    //         affine_wavefronts_compute_offsets_im_avx512(affine_wavefronts, &wavefront_set, lo, hi);
    //         break;
    //     case 1: // 01b
    //         affine_wavefronts_compute_offsets_dm_avx512(affine_wavefronts, &wavefront_set, lo, hi);
    //         break;
    //     case 0: // 00b
    //         affine_wavefronts_compute_offsets_m_avx512(affine_wavefronts, &wavefront_set, lo, hi);
    //         break;
    //     }
    // #else
    // printf("NONE");
    switch (kernel)
    {
    case 3: // 11b
        affine_wavefronts_compute_offsets_idm(affine_wavefronts, &wavefront_set, lo, hi);
        break;
    case 2: // 10b
        affine_wavefronts_compute_offsets_im(affine_wavefronts, &wavefront_set, lo, hi);
        break;
    case 1: // 01b
        affine_wavefronts_compute_offsets_dm(affine_wavefronts, &wavefront_set, lo, hi);
        break;
    case 0: // 00b
        affine_wavefronts_compute_offsets_m(affine_wavefronts, &wavefront_set, lo, hi);
        break;
    }
    // #endif

    WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_operations, hi - lo + 1);
    // DEBUG
#ifdef AFFINE_WAVEFRONT_DEBUG
    // Copy offsets base before extension (for display purposes)
    affine_wavefront_t *const mwavefront = affine_wavefronts->mwavefronts[score];
    if (mwavefront != NULL)
    {
        int k;
        for (k = mwavefront->lo; k <= mwavefront->hi; ++k)
        {
            mwavefront->offsets_base[k] = mwavefront->offsets[k];
        }
    }
#endif
}

/*
 * Computation using Wavefronts
 */
void affine_wavefronts_align(
    affine_wavefronts_t *const affine_wavefronts,
    const char *const pattern,
    const int pattern_length,
    const char *const text,
    const int text_length)
{

    struct timeval start_update_time, end_update_time;
    struct timeval start_lcq_time, end_lcq_time;

    // Init padded strings
    strings_padded_t *const strings_padded =
        strings_padded_new_rhomb(
            pattern, pattern_length, text, text_length,
            AFFINE_WAVEFRONT_PADDING, affine_wavefronts->mm_allocator);
    // Initialize wavefront
    affine_wavefront_initialize(affine_wavefronts);
    // Compute wavefronts for increasing score
    int score = 0;
    while (true)
    {
        double time_iter = 0;
        // Exact extend s-wavefront
        affine_wavefronts_extend_wavefront_packed(
            affine_wavefronts, strings_padded->pattern_padded, pattern_length,
            strings_padded->text_padded, text_length, score);
        // affine_wavefronts_extend_wavefront_packed(
        //     affine_wavefronts, strings_padded->pattern_padded, pattern_length,
        //     strings_padded->text_padded, text_length, score);
        time_iter = 0;
        // Exit condition
        if (affine_wavefront_end_reached(affine_wavefronts, pattern_length, text_length, score))
        {
            // Backtrace & check alignment reached
            affine_wavefronts_backtrace(
                affine_wavefronts, strings_padded->pattern_padded, pattern_length,
                strings_padded->text_padded, text_length, score);
            break;
        }
        // Update all wavefronts
        ++score; // Increase score

        affine_wavefronts_compute_wavefront(
            affine_wavefronts, strings_padded->pattern_padded, pattern_length,
            strings_padded->text_padded, text_length, score);
        // DEBUG
        // affine_wavefronts_debug_step(affine_wavefronts,pattern,text,score);
        WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_steps, 1);
    }
    // DEBUG
    // affine_wavefronts_debug_step(affine_wavefronts,pattern,text,score);
    WAVEFRONT_STATS_COUNTER_ADD(affine_wavefronts, wf_score, score); // STATS
    // Free
    strings_padded_delete(strings_padded);
}
