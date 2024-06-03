/*
 *                             The MIT License
 *
 * Wavefront Alignments Algorithms
 * Copyright (c) 2017 by Santiago Marco-Sola  <santiagomsola@gmail.com>
 *
 * This file is part of Wavefront Alignments Algorithms.
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
 * PROJECT: Wavefront Alignments Algorithms
 * AUTHOR(S): Santiago Marco-Sola <santiagomsola@gmail.com>
 * DESCRIPTION: WFA main algorithm
 */

#include "affine_wavefront_align.h"
#include "gap_affine/affine_wavefront_backtrace.h"
#include "gap_affine/affine_wavefront_display.h"
#include "gap_affine/affine_wavefront_extend.h"
#include "gap_affine/affine_wavefront_utils.h"
#include "utils/string_padded.h"
#include <immintrin.h>

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

// AVX2向量化实现部分

/***
 * 拓展 WFA，计算 M 矩阵的部分，使用 AVX2 向量化实现
 * 对于核心计算部分 (m_sub_lo <= (k) && (k) <= m_sub_hi) ? (m_sub_offsets[k] + 1) : ((-2147483647-1)/2)
 * 拆分为 if bool then a else b，计算 bool = (m_sub_lo <= (k) && (k) <= m_sub_hi)
 ***/
void affine_wavefronts_compute_offsets_m_avx2(
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
    const int batchSize = 8;
    // 设置向量，8个 sub_lo、sub_hi、1、-1、-MIN/2
    const __m256i loVec = _mm256_set1_epi32(m_sub_lo);
    const __m256i hiVec = _mm256_set1_epi32(m_sub_hi);
    const __m256i offsetAddition1 = _mm256_set1_epi32(1);
    const __m256i offsetMIN = _mm256_set1_epi32(-1073741824);
    __m256i k_vec;

    for (k = lo; k <= hi - batchSize + 1; k += batchSize)
    {
        // 设置向量单元：，m_sub_offsets[k:k+7]
        __m256i offsets_vec = _mm256_loadu_si256((__m256i *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = _mm256_add_epi32(offsets_vec, offsetAddition1);

        // 用于后续的 mask,当 k在[m_sub_lo, m_sub_hi]之中。
        // 正向掩码若是设为全 1，否则设为 0；反向掩码若否设为 1，是设为 0
        k_vec = _mm256_set_epi32(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        __m256i mask = _mm256_and_si256(_mm256_cmpgt_epi32(hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, loVec));

        // _mm256_blendv_epi8(a,b,mask)  if mask then b else a（掩码为 1则为 offsets_vec）
        __m256i result = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        // 存回数据
        _mm256_storeu_si256((__m256i *)&out_moffsets[k], result);
    }

    // 计算剩余部分
    for (; k <= hi; ++k)
    {
        out_moffsets[k] = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
    }
}

/***
 * 拓展 WFA，计算 M 矩阵的部分，使用 AVX2 向量化实现
 * 计算 m_gap/d_ext = (m_gap_lo <= (k + 1) && (k + 1) <= m_gap_hi)  ? (m_gap_offsets[k + 1]) : ((1073741824)
 * 计算     sub     = (m_sub_lo <= (k) && (k) <= m_sub_hi)          ? (m_sub_offsets[k] + 1) : ((1073741824)
 *
 ***/

void affine_wavefronts_compute_offsets_dm_avx2(
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
    const __m256i m_sub_loVec = _mm256_set1_epi32(m_sub_lo);
    const __m256i m_sub_hiVec = _mm256_set1_epi32(m_sub_hi);
    const __m256i m_gap_loVec = _mm256_set1_epi32(m_gap_lo);
    const __m256i m_gap_hiVec = _mm256_set1_epi32(m_gap_hi);
    const __m256i d_ext_loVec = _mm256_set1_epi32(d_ext_lo);
    const __m256i d_ext_hiVec = _mm256_set1_epi32(d_ext_hi);
    const __m256i offsetMIN = _mm256_set1_epi32(-1073741824);
    __m256i offsets_vec, k_vec, mask, result;

    // Compute score wavefronts
    int k;
    // AVX2每次处理的元素数目，256/32=8
    const int batchSize = 8;

    for (k = lo; k <= hi - batchSize + 1; k += batchSize)
    {

        // m_gap
        // m_gap_offsets[k+1:k+8]
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = _mm256_set_epi32(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_gap_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_gap_loVec));
        // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __m256i del_g = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        // d_ext
        offsets_vec = _mm256_loadu_si256((__m256i *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = _mm256_set_epi32(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(d_ext_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, d_ext_loVec));
        // sub <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __m256i del_d = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        // Update D
        __m256i del = _mm256_max_epi32(del_g, del_d);
        _mm256_storeu_si256((__m256i *)&out_doffsets[k], del);

        // m_sub
        // m_sub_offsets[k:k+7]
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = _mm256_add_epi32(offsets_vec, _mm256_set1_epi32(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = _mm256_set_epi32(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_sub_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_sub_loVec));
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __m256i sub = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        // Update D
        __m256i max_m = _mm256_max_epi32(del, sub);
        _mm256_storeu_si256((__m256i *)&out_moffsets[k], max_m);
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

void affine_wavefronts_compute_offsets_im_avx2(
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
    const __m256i m_sub_loVec = _mm256_set1_epi32(m_sub_lo);
    const __m256i m_sub_hiVec = _mm256_set1_epi32(m_sub_hi);
    const __m256i m_gap_loVec = _mm256_set1_epi32(m_gap_lo);
    const __m256i m_gap_hiVec = _mm256_set1_epi32(m_gap_hi);
    const __m256i i_ext_loVec = _mm256_set1_epi32(i_ext_lo);
    const __m256i i_ext_hiVec = _mm256_set1_epi32(i_ext_hi);
    const __m256i offsetMIN = _mm256_set1_epi32(-1073741824);
    __m256i offsets_vec, k_vec, mask, result;

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
    const int batchSize = 8;
    // 向量化部分
    for (; k <= hi - batchSize + 1; k += batchSize)
    {

        // Update I
        // m_gap_offsets[k-1:k+6]
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_gap_offsets[k - 1]);
        // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
        k_vec = _mm256_set_epi32(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_gap_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_gap_loVec));
        // sub <- if mask ? (m_gap_offsets[k-1]) : (-1073741824)
        __m256i ins_g = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = _mm256_loadu_si256((__m256i *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = _mm256_set_epi32(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(i_ext_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, i_ext_loVec));
        // sub <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __m256i ins_i = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        __m256i ins = _mm256_max_epi32(ins_g, ins_i);
        ins = _mm256_add_epi32(ins, _mm256_set1_epi32(1));
        _mm256_storeu_si256((__m256i *)&out_ioffsets[k], ins);

        // m_sub
        // m_sub_offsets[k:k+7]
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = _mm256_add_epi32(offsets_vec, _mm256_set1_epi32(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = _mm256_set_epi32(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_sub_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_sub_loVec));
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __m256i sub = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        // Update D
        __m256i max_m = _mm256_max_epi32(ins, sub);
        _mm256_storeu_si256((__m256i *)&out_moffsets[k], max_m);
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

void affine_wavefronts_compute_offsets_idm_avx2(
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
    const __m256i m_sub_loVec = _mm256_set1_epi32(m_sub_lo);
    const __m256i m_sub_hiVec = _mm256_set1_epi32(m_sub_hi);
    const __m256i m_gap_loVec = _mm256_set1_epi32(m_gap_lo);
    const __m256i m_gap_hiVec = _mm256_set1_epi32(m_gap_hi);
    const __m256i i_ext_loVec = _mm256_set1_epi32(i_ext_lo);
    const __m256i i_ext_hiVec = _mm256_set1_epi32(i_ext_hi);
    const __m256i d_ext_loVec = _mm256_set1_epi32(d_ext_lo);
    const __m256i d_ext_hiVec = _mm256_set1_epi32(d_ext_hi);
    const __m256i offsetMIN = _mm256_set1_epi32(-1073741824);
    __m256i offsets_vec, k_vec, mask;
    // AVX2每次处理的元素数目，256/32=8
    const int batchSize = 8;

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
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_gap_offsets[k - 1]);
        // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
        k_vec = _mm256_set_epi32(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_gap_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_gap_loVec));
        // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __m256i ins_g = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);
        // ins_i
        offsets_vec = _mm256_loadu_si256((__m256i *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = _mm256_set_epi32(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(i_ext_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, i_ext_loVec));
        // ins_i <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __m256i ins_i = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        __m256i ins = _mm256_max_epi32(ins_g, ins_i);
        ins = _mm256_add_epi32(ins, _mm256_set1_epi32(1));
        _mm256_storeu_si256((__m256i *)&out_ioffsets[k], ins);

        // Update D
        // del_g
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = _mm256_set_epi32(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_gap_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_gap_loVec));
        // del_g <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __m256i del_g = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = _mm256_loadu_si256((__m256i *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = _mm256_set_epi32(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(d_ext_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, d_ext_loVec));
        // del_d <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __m256i del_d = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        __m256i del = _mm256_max_epi32(del_g, del_d);
        _mm256_storeu_si256((__m256i *)&out_doffsets[k], del);

        // Update M
        // m_sub_offsets[k:k+7]
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = _mm256_add_epi32(offsets_vec, _mm256_set1_epi32(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = _mm256_set_epi32(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_sub_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_sub_loVec));
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __m256i sub = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        __m256i max_m = _mm256_max_epi32(del, sub);
        max_m = _mm256_max_epi32(max_m, ins);
        _mm256_storeu_si256((__m256i *)&out_moffsets[k], max_m);
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
        __m256i m_gapi_value = _mm256_loadu_si256((__m256i *)&m_gap_offsets[k - 1]);
        __m256i i_ext_value = _mm256_loadu_si256((__m256i *)&i_ext_offsets[k - 1]);
        __m256i ins = _mm256_max_epi32(i_ext_value, m_gapi_value);
        ins = _mm256_add_epi32(ins, _mm256_set1_epi32(1));
        _mm256_storeu_si256((__m256i *)&out_ioffsets[k], ins);
        // Update D
        __m256i m_gapd_value = _mm256_loadu_si256((__m256i *)&m_gap_offsets[k + 1]);
        __m256i d_ext_value = _mm256_loadu_si256((__m256i *)&d_ext_offsets[k + 1]);
        __m256i del = _mm256_max_epi32(m_gapd_value, d_ext_value);
        _mm256_storeu_si256((__m256i *)&out_doffsets[k], del);
        // Update M
        __m256i sub = _mm256_loadu_si256((__m256i *)&m_sub_offsets[k]);
        sub = _mm256_add_epi32(sub, _mm256_set1_epi32(1));
        sub = _mm256_max_epi32(sub, del);
        sub = _mm256_max_epi32(sub, ins);
        _mm256_storeu_si256((__m256i *)&out_moffsets[k], sub);
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
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_gap_offsets[k - 1]);
        // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
        k_vec = _mm256_set_epi32(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_gap_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_gap_loVec));
        // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __m256i ins_g = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);
        // ins_i
        offsets_vec = _mm256_loadu_si256((__m256i *)&i_ext_offsets[k - 1]);
        // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
        k_vec = _mm256_set_epi32(k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(i_ext_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, i_ext_loVec));
        // ins_i <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
        __m256i ins_i = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        __m256i ins = _mm256_max_epi32(ins_g, ins_i);
        ins = _mm256_add_epi32(ins, _mm256_set1_epi32(1));
        _mm256_storeu_si256((__m256i *)&out_ioffsets[k], ins);

        // Update D
        // del_g
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_gap_offsets[k + 1]);
        // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
        k_vec = _mm256_set_epi32(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_gap_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_gap_loVec));
        // del_g <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
        __m256i del_g = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);
        // d_ext
        offsets_vec = _mm256_loadu_si256((__m256i *)&d_ext_offsets[k + 1]);
        // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
        k_vec = _mm256_set_epi32(k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(d_ext_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, d_ext_loVec));
        // del_d <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
        __m256i del_d = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        __m256i del = _mm256_max_epi32(del_g, del_d);
        _mm256_storeu_si256((__m256i *)&out_doffsets[k], del);

        // Update M
        // m_sub_offsets[k:k+7]
        offsets_vec = _mm256_loadu_si256((__m256i *)&m_sub_offsets[k]);
        //  m_sub_offsets[k] + 1
        offsets_vec = _mm256_add_epi32(offsets_vec, _mm256_set1_epi32(1));
        // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
        k_vec = _mm256_set_epi32(k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
        mask = _mm256_and_si256(_mm256_cmpgt_epi32(m_sub_hiVec, k_vec), _mm256_cmpgt_epi32(k_vec, m_sub_loVec));
        // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
        __m256i sub = _mm256_blendv_epi8(offsetMIN, offsets_vec, mask);

        __m256i max_m = _mm256_max_epi32(del, sub);
        max_m = _mm256_max_epi32(max_m, ins);
        _mm256_storeu_si256((__m256i *)&out_moffsets[k], max_m);
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

// // AVX512
// void affine_wavefronts_compute_offsets_m_avx512(
//     affine_wavefronts_t *const affine_wavefronts,
//     const affine_wavefront_set *const wavefront_set,
//     const int lo, const int hi)
// {
//     // Parameters
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
//     awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

//     // Compute score wavefronts
//     int k;
//     // AVX512每次处理的元素数目，512/32=16
//     const int batchSize = 16;
//     // 设置向量，8个 sub_lo、sub_hi、1、-1、-MIN/2
//     const __m512i loVec = _mm512_set1_epi32(m_sub_lo);
//     const __m512i hiVec = _mm512_set1_epi32(m_sub_hi);
//     const __m512i offsetAddition1 = _mm512_set1_epi32(1);
//     const __m512i offsetMIN = _mm512_set1_epi32(-1073741824);
//     __m512i k_vec;
//     __mmask16 mask;

//     for (k = lo; k <= hi - batchSize + 1; k += batchSize)
//     {
//         // 设置向量单元：，m_sub_offsets[k:k+7]
//         __m512i offsets_vec = _mm512_loadu_si512((__m512i *)&m_sub_offsets[k]);
//         //  m_sub_offsets[k] + 1
//         offsets_vec = _mm512_add_epi32(offsets_vec, offsetAddition1);
//         // 用于后续的 mask,当 k在[m_sub_lo, m_sub_hi]之中。
//         // 正向掩码若是设为全 1，否则设为 0；反向掩码若否设为 1，是设为 0
//         k_vec = _mm512_set_epi32(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
//         mask = (_mm512_cmpgt_epi32_mask(hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, loVec));
//         // _mm512_mask_blend_epi32(mask,a,b)  if mask then b else a（掩码为 1则为 offsets_vec）
//         __m512i result = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         // 存回数据
//         _mm512_storeu_si512((__m512i *)&out_moffsets[k], result);
//     }

//     // 计算剩余部分
//     for (; k <= hi; ++k)
//     {
//         out_moffsets[k] = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
//     }
// }

// void affine_wavefronts_compute_offsets_dm_avx512(
//     affine_wavefronts_t *const affine_wavefronts,
//     const affine_wavefront_set *const wavefront_set,
//     const int lo, const int hi)
// {
//     // Parameters
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_dwavefront_ext, d_ext);
//     awf_offset_t *const out_doffsets = wavefront_set->out_dwavefront->offsets;
//     awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

//     // 设置向量
//     const __m512i m_sub_loVec = _mm512_set1_epi32(m_sub_lo);
//     const __m512i m_sub_hiVec = _mm512_set1_epi32(m_sub_hi);
//     const __m512i m_gap_loVec = _mm512_set1_epi32(m_gap_lo);
//     const __m512i m_gap_hiVec = _mm512_set1_epi32(m_gap_hi);
//     const __m512i d_ext_loVec = _mm512_set1_epi32(d_ext_lo);
//     const __m512i d_ext_hiVec = _mm512_set1_epi32(d_ext_hi);
//     const __m512i offsetMIN = _mm512_set1_epi32(-1073741824);
//     __m512i offsets_vec, k_vec, result;
//     __mmask16 mask;

//     // Compute score wavefronts
//     int k;
//     // AVX512每次处理的元素数目，512/32=16
//     const int batchSize = 16;
//     // 向量化部分
//     for (k = lo; k <= hi - batchSize + 1; k += batchSize)
//     {

//         // Update D
//         // m_gap_offsets[k+1:k+16]
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_gap_offsets[k + 1]);
//         // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
//         k_vec = _mm512_set_epi32(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
//         mask = (_mm512_cmpgt_epi32_mask(m_gap_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_gap_loVec));
//         // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
//         __m512i del_g = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);
//         // d_ext
//         offsets_vec = _mm512_loadu_si512((__m512i *)&d_ext_offsets[k + 1]);
//         // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
//         k_vec = _mm512_set_epi32(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
//         mask = (_mm512_cmpgt_epi32_mask(d_ext_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, d_ext_loVec));
//         // sub <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
//         __m512i del_d = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i del = _mm512_max_epi32(del_g, del_d);
//         _mm512_storeu_si512((__m512i *)&out_doffsets[k], del);

//         // Update M
//         // m_sub_offsets[k:k+15]
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_sub_offsets[k]);
//         //  m_sub_offsets[k] + 1
//         offsets_vec = _mm512_add_epi32(offsets_vec, _mm512_set1_epi32(1));
//         // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
//         k_vec = _mm512_set_epi32(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
//         mask = (_mm512_cmpgt_epi32_mask(m_sub_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_sub_loVec));
//         // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
//         __m512i sub = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i max_m = _mm512_max_epi32(del, sub);
//         _mm512_storeu_si512((__m512i *)&out_moffsets[k], max_m);
//     }

//     // 计算剩余部分
//     for (; k <= hi; ++k)
//     {
//         // Update D
//         const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
//         const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
//         const awf_offset_t del = MAX(del_g, del_d);
//         out_doffsets[k] = del;
//         // Update M
//         const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
//         out_moffsets[k] = MAX(del, sub);
//     }
// }

// void affine_wavefronts_compute_offsets_im_avx512(
//     affine_wavefronts_t *const affine_wavefronts,
//     const affine_wavefront_set *const wavefront_set,
//     const int lo, const int hi)
// {
//     // Parameters
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_iwavefront_ext, i_ext);
//     awf_offset_t *const out_ioffsets = wavefront_set->out_iwavefront->offsets;
//     awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;

//     // 设置向量
//     const __m512i m_sub_loVec = _mm512_set1_epi32(m_sub_lo);
//     const __m512i m_sub_hiVec = _mm512_set1_epi32(m_sub_hi);
//     const __m512i m_gap_loVec = _mm512_set1_epi32(m_gap_lo);
//     const __m512i m_gap_hiVec = _mm512_set1_epi32(m_gap_hi);
//     const __m512i i_ext_loVec = _mm512_set1_epi32(i_ext_lo);
//     const __m512i i_ext_hiVec = _mm512_set1_epi32(i_ext_hi);
//     const __m512i offsetMIN = _mm512_set1_epi32(-1073741824);
//     __m512i offsets_vec, k_vec, result;
//     __mmask16 mask;

//     // Compute score wavefronts
//     int k = lo;

//     // 第一次计算
//     // Update I
//     const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
//     const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
//     const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
//     out_ioffsets[k] = ins;
//     // Update M
//     const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
//     out_moffsets[k] = MAX(ins, sub);
//     k++;

//     // AVX512每次处理的元素数目，512/32=16
//     const int batchSize = 16;
//     // 向量化部分
//     for (; k <= hi - batchSize + 1; k += batchSize)
//     {

//         // Update I
//         // m_gap_offsets[k-1:k+14]
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_gap_offsets[k - 1]);
//         // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
//         k_vec = _mm512_set_epi32(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
//                                  k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
//         mask = (_mm512_cmpgt_epi32_mask(m_gap_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_gap_loVec));
//         // sub <- if mask ? (m_gap_offsets[k-1]) : (-1073741824)
//         __m512i ins_g = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);
//         // d_ext
//         offsets_vec = _mm512_loadu_si512((__m512i *)&i_ext_offsets[k - 1]);
//         // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
//         k_vec = _mm512_set_epi32(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
//                                  k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
//         mask = (_mm512_cmpgt_epi32_mask(i_ext_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, i_ext_loVec));
//         // sub <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
//         __m512i ins_i = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i ins = _mm512_max_epi32(ins_g, ins_i);
//         ins = _mm512_add_epi32(ins, _mm512_set1_epi32(1));
//         _mm512_storeu_si512((__m512i *)&out_ioffsets[k], ins);

//         // Update M
//         // m_sub_offsets[k:k+15]
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_sub_offsets[k]);
//         //  m_sub_offsets[k] + 1
//         offsets_vec = _mm512_add_epi32(offsets_vec, _mm512_set1_epi32(1));
//         // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
//         k_vec = _mm512_set_epi32(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
//         mask = (_mm512_cmpgt_epi32_mask(m_sub_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_sub_loVec));
//         // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
//         __m512i sub = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i max_m = _mm512_max_epi32(ins, sub);
//         _mm512_storeu_si512((__m512i *)&out_moffsets[k], max_m);
//     }

//     // 计算剩余部分
//     for (; k <= hi; ++k)
//     {
//         // Update I
//         const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
//         const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
//         const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
//         out_ioffsets[k] = ins;
//         // Update M
//         const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
//         out_moffsets[k] = MAX(ins, sub);
//     }
// }

// void affine_wavefronts_compute_offsets_idm_avx512(
//     affine_wavefronts_t *const affine_wavefronts,
//     const affine_wavefront_set *const wavefront_set,
//     const int lo,
//     const int hi)
// {
//     // Parameters
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_sub, m_sub);
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_mwavefront_gap, m_gap);
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_iwavefront_ext, i_ext);
//     AFFINE_WAVEFRONT_DECLARE(wavefront_set->in_dwavefront_ext, d_ext);
//     awf_offset_t *const out_ioffsets = wavefront_set->out_iwavefront->offsets;
//     awf_offset_t *const out_doffsets = wavefront_set->out_dwavefront->offsets;
//     awf_offset_t *const out_moffsets = wavefront_set->out_mwavefront->offsets;
//     // Compute loop peeling offset (min_hi)
//     int min_hi = wavefront_set->in_mwavefront_sub->hi;
//     if (!wavefront_set->in_mwavefront_gap->null && min_hi > wavefront_set->in_mwavefront_gap->hi - 1)
//         min_hi = wavefront_set->in_mwavefront_gap->hi - 1;
//     if (!wavefront_set->in_iwavefront_ext->null && min_hi > wavefront_set->in_iwavefront_ext->hi + 1)
//         min_hi = wavefront_set->in_iwavefront_ext->hi + 1;
//     if (!wavefront_set->in_dwavefront_ext->null && min_hi > wavefront_set->in_dwavefront_ext->hi - 1)
//         min_hi = wavefront_set->in_dwavefront_ext->hi - 1;
//     // Compute loop peeling offset (max_lo)
//     int max_lo = wavefront_set->in_mwavefront_sub->lo;
//     if (!wavefront_set->in_mwavefront_gap->null && max_lo < wavefront_set->in_mwavefront_gap->lo + 1)
//         max_lo = wavefront_set->in_mwavefront_gap->lo + 1;
//     if (!wavefront_set->in_iwavefront_ext->null && max_lo < wavefront_set->in_iwavefront_ext->lo + 1)
//         max_lo = wavefront_set->in_iwavefront_ext->lo + 1;
//     if (!wavefront_set->in_dwavefront_ext->null && max_lo < wavefront_set->in_dwavefront_ext->lo - 1)
//         max_lo = wavefront_set->in_dwavefront_ext->lo - 1;
//     // Compute score wavefronts (prologue)

//     // 设置向量
//     const __m512i m_sub_loVec = _mm512_set1_epi32(m_sub_lo);
//     const __m512i m_sub_hiVec = _mm512_set1_epi32(m_sub_hi);
//     const __m512i m_gap_loVec = _mm512_set1_epi32(m_gap_lo);
//     const __m512i m_gap_hiVec = _mm512_set1_epi32(m_gap_hi);
//     const __m512i i_ext_loVec = _mm512_set1_epi32(i_ext_lo);
//     const __m512i i_ext_hiVec = _mm512_set1_epi32(i_ext_hi);
//     const __m512i d_ext_loVec = _mm512_set1_epi32(d_ext_lo);
//     const __m512i d_ext_hiVec = _mm512_set1_epi32(d_ext_hi);
//     const __m512i offsetMIN = _mm512_set1_epi32(-1073741824);
//     __m512i offsets_vec, k_vec;
//     __mmask16 mask;
//     // AVX2每次处理的元素数目，256/32=8
//     const int batchSize = 8;

//     int k = lo;
//     // 第一次计算
//     // Compute score wavefronts
//     {
//         // Update I
//         const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
//         const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
//         const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
//         out_ioffsets[k] = ins;
//         // Update D
//         const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
//         const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
//         const awf_offset_t del = MAX(del_g, del_d);
//         out_doffsets[k] = del;
//         // Update M
//         const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
//         out_moffsets[k] = MAX(del, MAX(sub, ins));
//         k++;
//     }
//     // 向量化部分
//     for (; k < max_lo - batchSize + 1; k += batchSize)
//     {
//         // Update I
//         // m_gap_offsets[k-1:k+14]
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_gap_offsets[k - 1]);
//         // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
//         k_vec = _mm512_set_epi32(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
//                                  k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
//         mask = (_mm512_cmpgt_epi32_mask(m_gap_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_gap_loVec));
//         // sub <- if mask ? (m_gap_offsets[k-1]) : (-1073741824)
//         __m512i ins_g = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);
//         // d_ext
//         offsets_vec = _mm512_loadu_si512((__m512i *)&i_ext_offsets[k - 1]);
//         // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
//         k_vec = _mm512_set_epi32(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
//                                  k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
//         mask = (_mm512_cmpgt_epi32_mask(i_ext_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, i_ext_loVec));
//         // sub <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
//         __m512i ins_i = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i ins = _mm512_max_epi32(ins_g, ins_i);
//         ins = _mm512_add_epi32(ins, _mm512_set1_epi32(1));
//         _mm512_storeu_si512((__m512i *)&out_ioffsets[k], ins);

//         // Update D
//         // m_gap_offsets[k+1:k+16]
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_gap_offsets[k + 1]);
//         // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
//         k_vec = _mm512_set_epi32(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
//         mask = (_mm512_cmpgt_epi32_mask(m_gap_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_gap_loVec));
//         // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
//         __m512i del_g = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);
//         // d_ext
//         offsets_vec = _mm512_loadu_si512((__m512i *)&d_ext_offsets[k + 1]);
//         // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
//         k_vec = _mm512_set_epi32(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
//         mask = (_mm512_cmpgt_epi32_mask(d_ext_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, d_ext_loVec));
//         // sub <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
//         __m512i del_d = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i del = _mm512_max_epi32(del_g, del_d);
//         _mm512_storeu_si512((__m512i *)&out_doffsets[k], del);

//         // Update M
//         // m_sub_offsets[k:k+7]
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_sub_offsets[k]);
//         //  m_sub_offsets[k] + 1
//         offsets_vec = _mm512_add_epi32(offsets_vec, _mm512_set1_epi32(1));
//         // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
//         k_vec = _mm512_set_epi32(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
//         mask = (_mm512_cmpgt_epi32_mask(m_sub_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_sub_loVec));
//         // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
//         __m512i sub = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i max_m = _mm512_max_epi32(del, sub);
//         max_m = _mm512_max_epi32(max_m, ins);
//         _mm512_storeu_si512((__m512i *)&out_moffsets[k], max_m);
//     }
//     // 计算剩余部分
//     for (; k < max_lo; ++k)
//     {
//         // Update I
//         const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
//         const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
//         const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
//         out_ioffsets[k] = ins;
//         // Update D
//         const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
//         const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
//         const awf_offset_t del = MAX(del_g, del_d);
//         out_doffsets[k] = del;
//         // Update M
//         const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
//         out_moffsets[k] = MAX(del, MAX(sub, ins));
//     }

//     // Compute score wavefronts (core)

//     // 向量化部分
//     for (; k <= min_hi - batchSize + 1; k += batchSize)
//     {
//         // Update I
//         __m512i m_gapi_value = _mm512_loadu_si512((__m512i *)&m_gap_offsets[k - 1]);
//         __m512i i_ext_value = _mm512_loadu_si512((__m512i *)&i_ext_offsets[k - 1]);
//         __m512i ins = _mm512_max_epi32(i_ext_value, m_gapi_value);
//         ins = _mm512_add_epi32(ins, _mm512_set1_epi32(1));
//         _mm512_storeu_si512((__m512i *)&out_ioffsets[k], ins);
//         // Update D
//         __m512i m_gapd_value = _mm512_loadu_si512((__m512i *)&m_gap_offsets[k + 1]);
//         __m512i d_ext_value = _mm512_loadu_si512((__m512i *)&d_ext_offsets[k + 1]);
//         __m512i del = _mm512_max_epi32(m_gapd_value, d_ext_value);
//         _mm512_storeu_si512((__m512i *)&out_doffsets[k], del);
//         // Update M
//         __m512i sub = _mm512_loadu_si512((__m512i *)&m_sub_offsets[k]);
//         sub = _mm512_add_epi32(sub, _mm512_set1_epi32(1));
//         sub = _mm512_max_epi32(sub, del);
//         sub = _mm512_max_epi32(sub, ins);
//         _mm512_storeu_si512((__m512i *)&out_moffsets[k], sub);
//     }
//     for (; k <= min_hi; ++k)
//     {
//         // Update I
//         const awf_offset_t m_gapi_value = m_gap_offsets[k - 1];
//         const awf_offset_t i_ext_value = i_ext_offsets[k - 1];
//         const awf_offset_t ins = MAX(m_gapi_value, i_ext_value) + 1;
//         out_ioffsets[k] = ins;
//         // Update D
//         const awf_offset_t m_gapd_value = m_gap_offsets[k + 1];
//         const awf_offset_t d_ext_value = d_ext_offsets[k + 1];
//         const awf_offset_t del = MAX(m_gapd_value, d_ext_value);
//         out_doffsets[k] = del;
//         // Update M
//         const awf_offset_t sub = m_sub_offsets[k] + 1;
//         out_moffsets[k] = MAX(del, MAX(sub, ins));
//     }

//     // Compute score wavefronts (epilogue)
//     for (; k <= min_hi - batchSize + 1; k += batchSize)
//     {
//         // Update I
//         // ins_g
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_gap_offsets[k - 1]);
//         // mask <- (m_gap_lo <= (k-1) && (k-1) <= m_gap_hi)
//         k_vec = _mm512_set_epi32(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
//                                  k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
//         mask = (_mm512_cmpgt_epi32_mask(m_gap_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_gap_loVec));
//         // sub <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
//         __m512i ins_g = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);
//         // ins_i
//         offsets_vec = _mm512_loadu_si512((__m512i *)&i_ext_offsets[k - 1]);
//         // mask <- (d_ext_lo <= (k-1) && (k-1) <= d_ext_hi)
//         k_vec = _mm512_set_epi32(k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
//                                  k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k, k - 1);
//         mask = (_mm512_cmpgt_epi32_mask(i_ext_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, i_ext_loVec));
//         // ins_i <- if mask ? (i_ext_offsets[k-1]) : (-1073741824)
//         __m512i ins_i = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i ins = _mm512_max_epi32(ins_g, ins_i);
//         ins = _mm512_add_epi32(ins, _mm512_set1_epi32(1));
//         _mm512_storeu_si512((__m512i *)&out_ioffsets[k], ins);

//         // Update D
//         // del_g
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_gap_offsets[k + 1]);
//         // mask <- (m_gap_lo <= (k+1) && (k+1) <= m_gap_hi)
//         k_vec = _mm512_set_epi32(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
//         mask = (_mm512_cmpgt_epi32_mask(m_gap_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_gap_loVec));
//         // del_g <- if mask ? (m_gap_offsets[k+1]) : (-1073741824)
//         __m512i del_g = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);
//         // d_ext
//         offsets_vec = _mm512_loadu_si512((__m512i *)&d_ext_offsets[k + 1]);
//         // mask <- (d_ext_lo <= (k+1) && (k+1) <= d_ext_hi)
//         k_vec = _mm512_set_epi32(k + 16, k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1);
//         mask = (_mm512_cmpgt_epi32_mask(d_ext_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, d_ext_loVec));
//         // del_d <- if mask ? (d_ext_offsets[k+1]) : (-1073741824)
//         __m512i del_d = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i del = _mm512_max_epi32(del_g, del_d);
//         _mm512_storeu_si512((__m512i *)&out_doffsets[k], del);

//         // Update M
//         // m_sub_offsets[k:k+7]
//         offsets_vec = _mm512_loadu_si512((__m512i *)&m_sub_offsets[k]);
//         //  m_sub_offsets[k] + 1
//         offsets_vec = _mm512_add_epi32(offsets_vec, _mm512_set1_epi32(1));
//         // mask <- (m_sub_lo <= (k) && (k) <= m_sub_hi)
//         k_vec = _mm512_set_epi32(k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9,
//                                  k + 8, k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
//         mask = (_mm512_cmpgt_epi32_mask(m_sub_hiVec, k_vec)) & (_mm512_cmpgt_epi32_mask(k_vec, m_sub_loVec));
//         // sub <- if mask ? (m_sub_offsets[k] + 1) : (-1073741824)
//         __m512i sub = _mm512_mask_blend_epi32(mask, offsetMIN, offsets_vec);

//         __m512i max_m = _mm512_max_epi32(del, sub);
//         max_m = _mm512_max_epi32(max_m, ins);
//         _mm512_storeu_si512((__m512i *)&out_moffsets[k], max_m);
//     }
//     // 计算剩余部分
//     for (; k <= hi; ++k)
//     {
//         // Update I
//         const awf_offset_t ins_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k - 1, m_gap_offsets[k - 1]);
//         const awf_offset_t ins_i = AFFINE_WAVEFRONT_COND_FETCH(i_ext, k - 1, i_ext_offsets[k - 1]);
//         const awf_offset_t ins = MAX(ins_g, ins_i) + 1;
//         out_ioffsets[k] = ins;
//         // Update D
//         const awf_offset_t del_g = AFFINE_WAVEFRONT_COND_FETCH(m_gap, k + 1, m_gap_offsets[k + 1]);
//         const awf_offset_t del_d = AFFINE_WAVEFRONT_COND_FETCH(d_ext, k + 1, d_ext_offsets[k + 1]);
//         const awf_offset_t del = MAX(del_g, del_d);
//         out_doffsets[k] = del;
//         // Update M
//         const awf_offset_t sub = AFFINE_WAVEFRONT_COND_FETCH(m_sub, k, m_sub_offsets[k] + 1);
//         out_moffsets[k] = MAX(del, MAX(sub, ins));
//     }
// }

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
    // printf("case %d\n", kernel);
#ifdef ENABLE_AVX2
    // printf("AVX2");
    switch (kernel)
    {
    case 3: // 11b
        affine_wavefronts_compute_offsets_idm_avx2(affine_wavefronts, &wavefront_set, lo, hi);
        break;
    case 2: // 10b
        affine_wavefronts_compute_offsets_im_avx2(affine_wavefronts, &wavefront_set, lo, hi);
        break;
    case 1: // 01b
        affine_wavefronts_compute_offsets_dm_avx2(affine_wavefronts, &wavefront_set, lo, hi);
        break;
    case 0: // 00b
        affine_wavefronts_compute_offsets_m_avx2(affine_wavefronts, &wavefront_set, lo, hi);
        break;
    }
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
#else
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
#endif

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
