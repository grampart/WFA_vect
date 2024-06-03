#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "gap_affine/affine_wavefront_align.h"
int main(int argc, char *argv[])
{
    FILE *file;
    char *pattern, *text;
    char *path;
    int opt;
    size_t str_length = 10000; // 默认read长度
    size_t len = 0;

    struct timeval start_time, end_time;

    while ((opt = getopt(argc, argv, "p:l:")) != -1)
    {
        switch (opt)
        {
        case 'p':
            path = optarg;
            break;
        case 'l':
            str_length = atoi(optarg);
            break;
        default:
            fprintf(stderr, "错误用法: %s\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if ((file = fopen(path, "r")) == NULL)
    {
        printf("file %s does not exist.\n", path);
        return -1;
    }

    len = str_length * 1.3 + 1;

    // 分配空间
    pattern = (char *)malloc((len) * sizeof(char));
    text = (char *)malloc((len) * sizeof(char));

    gettimeofday(&start_time, NULL);

    fgets(pattern, len, file);
    fgets(text, len, file);

    memmove(pattern, pattern + 1, strlen(pattern));
    memmove(text, text + 1, strlen(text));
    pattern[strlen(pattern) - 1] = '\0';
    text[strlen(text) - 1] = '\0';

    mm_allocator_t *const mm_allocator = mm_allocator_new(BUFFER_SIZE_8M);

    affine_penalties_t affine_penalties = {
        .match = 0,
        .mismatch = 4,
        .gap_opening = 6,
        .gap_extension = 2,
    };

    affine_wavefronts_t *affine_wavefronts = affine_wavefronts_new_complete(
        strlen(pattern), strlen(text), &affine_penalties, NULL, mm_allocator);

    affine_wavefronts_align(
        affine_wavefronts, pattern, strlen(pattern), text, strlen(text));

#ifdef ENABLE_AVX2
    fprintf(stderr, "AVX2 vectorized\n");
#elif ENABLE_AVX512
    fprintf(stderr, "AVX512 vectorized\n");
#else
    fprintf(stderr, "not vectorized\n");
#endif
    const int score = edit_cigar_score_gap_affine(
        &affine_wavefronts->edit_cigar, &affine_penalties);
    fprintf(stderr, "  PATTERN  %s\n", pattern);
    fprintf(stderr, "  TEXT     %s\n", text);
    fprintf(stderr, "  SCORE COMPUTED %d\t", score);
    edit_cigar_print_pretty(stderr,
                            pattern, strlen(pattern), text, strlen(text),
                            &affine_wavefronts->edit_cigar, mm_allocator);

    affine_wavefronts_delete(affine_wavefronts);
    mm_allocator_delete(mm_allocator);
    return 0;
}
