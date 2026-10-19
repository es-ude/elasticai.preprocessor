#ifndef RESCALER_TEMPLATE_H
#define RESCALER_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>
#include <math.h>


#ifndef DEF_CALC_RESCALER_QUANT
#define DEF_CALC_RESCALER_QUANT(id, input_type, output_type) \
output_type rescaling_ ## id( \
        input_type data, float scale, int64_t zero_point, \
        int64_t min_int, int64_t max_int) { \
    int64_t ival = (int64_t)rintf((float)data / scale) + zero_point; \
    if (ival < min_int) ival = min_int; \
    if (ival > max_int) ival = max_int; \
    return (output_type)ival; \
}
#endif


#ifndef DEF_RESCALER_QUANT_IMPL
#define DEF_RESCALER_QUANT_IMPL(id, input_type, output_type, total_bits, is_signed, scale_val, zero_point_val) \
static DEF_CALC_RESCALER_QUANT(id, input_type, output_type) \
bool rescaling_data_ ## id(input_type data, output_type *out) { \
    static const float   adc_scale      = (float)(scale_val); \
    static const int64_t adc_zero_point = (int64_t)(zero_point_val); \
    static const int64_t adc_min        = (is_signed) \
                                          ? -((int64_t)1 << ((total_bits) - 1)) \
                                          : 0LL; \
    static const int64_t adc_max        = (is_signed) \
                                          ? (((int64_t)1 << ((total_bits) - 1)) - 1LL) \
                                          : (((int64_t)1 << (total_bits)) - 1LL); \
    *out = rescaling_ ## id(data, adc_scale, adc_zero_point, adc_min, adc_max); \
    return true; \
}
#endif


#ifndef DEF_RESCALER_QUANT_PROTO
#define DEF_RESCALER_QUANT_PROTO(id, input_type, output_type) \
bool rescaling_data_ ## id(input_type data, output_type *out);
#endif


#endif /* RESCALER_TEMPLATE_H */