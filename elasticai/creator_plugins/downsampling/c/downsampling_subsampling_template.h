#ifndef DOWNSAMPLING_SUBSAMPLING_TEMPLATE_H
#define DOWNSAMPLING_SUBSAMPLING_TEMPLATE_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    size_t output_len;
    size_t pos;
    void *out_arr;
} DoSubOut;

#ifndef DEF_DOWNSAMPLING_SUBSAMPLING
#define DEF_DOWNSAMPLING_SUBSAMPLING(id, input_type) \
bool calc_next_datum_do_subsampling_ ## id(input_type data, DoSubOut *out) { \
    size_t do_len = out->output_len; \
    size_t do_pos = out->pos; \
    input_type *do_out_arr = (input_type *) out->out_arr; \
    do_out_arr[do_pos] = data; \
    do_pos++; \
    if (do_pos >= do_len) { \
        out->pos = 0; \
        return true; \
    } \
    out->pos = do_pos; \
    return false; \
}
#endif//DEF_DOWNSAMPLING_SUBSAMPLING_IMPL

#ifndef DEF_NEW_DO_SUBSAMPLING_OUTPUT_ARRAY_IMPL
#define DEF_NEW_DO_SUBSAMPLING_OUTPUT_ARRAY_IMPL(id, input_type, dsr) \
static DEF_DOWNSAMPLING_SUBSAMPLING(id, input_type) \
bool calc_do_subsampling_ ## id (input_type data, input_type *out) { \
    static input_type do_out_arr[dsr] = {0}; \
    static DoSubOut output = { \
        .output_len = dsr, \
        .pos = 0, \
        .out_arr = do_out_arr, \
    }; \
    if (calc_next_datum_do_subsampling_ ## id (data, &(output))) { \
        for (size_t i = 0; i < dsr; i++) \
        { \
            out[i] = ((input_type *)output.out_arr)[i]; \
        } \
        return true; \
    } \
    return false; \
}
#endif//DEF_NEW_DO_SUBSAMPLING_OUTPUT_ARRAY_IMPL

#ifndef DEF_DOWNSAMPLING_SUBSAMPLING_PROTO
#define DEF_DOWNSAMPLING_SUBSAMPLING_PROTO(id, input_type) \
bool calc_do_subsampling_ ## id (input_type data, input_type *out);
#endif//DEF_DOWNSAMPLING_SUBSAMPLING_PROTO

#endif//DOWNSAMPLING_SUBSAMPLING_TEMPLATE_H
