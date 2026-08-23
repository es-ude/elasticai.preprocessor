#ifndef THRESHOLDING_CONSTANT_TEMPLATE_H
#define THRESHOLDING_CONSTANT_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>

#ifndef DEF_CONSTANT_THR_IMPL
#define DEF_CONSTANT_THR_IMPL(id, input_type, thr_const, gain_val) \
bool calc_thresholding_constant_ ## id(input_type data, input_type *out) { \
    static const input_type threshold = (input_type)(thr_const); \
    static const float thr_gain = (float)(gain_val); \
    *out = (input_type) (thr_gain * threshold); \
    return true; \
}
#endif//DEF_CONSTANT_THR_IMPL

#ifndef DEF_CONSTANT_THR_PROTO
#define DEF_CONSTANT_THR_PROTO(id, input_type) \
bool calc_thresholding_constant_ ## id(input_type data, input_type *out);
#endif//DEF_CONSTANT_THR_PROTO

#endif//THRESHOLDING_CONSTANT_TEMPLATE_H
