#ifndef ADC_TEMPLATE_H
#define ADC_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>
#include <math.h>


#ifndef DEF_CALC_ADC_QUANT
#define DEF_CALC_ADC_QUANT(id, output_type) \
output_type calc_next_datum_adc_ ## id( \
        float data, float vneg, float vpos, \
        float lsb, int64_t min_int, int64_t max_int) { \
    float clamped = data < vneg ? vneg : (data > vpos ? vpos : data); \
    int64_t steps = (int64_t)roundf((clamped - vneg) / lsb); \
    int64_t ival  = steps + min_int; \
    if (ival < min_int) ival = min_int; \
    if (ival > max_int) ival = max_int; \
    return (output_type)ival; \
}
#endif


#ifndef DEF_ADC_QUANT_IMPL
#define DEF_ADC_QUANT_IMPL(id, output_type, total_bits, is_signed, vneg_val, vpos_val) \
static DEF_CALC_ADC_QUANT(id, output_type) \
bool calc_adc_ ## id(float data, output_type *out) { \
    static const float   adc_vneg  = (float)(vneg_val); \
    static const float   adc_vpos  = (float)(vpos_val); \
    static const float   adc_lsb   = ((float)(vpos_val) - (float)(vneg_val)) \
                                     / (float)((int64_t)1 << (total_bits)); \
    static const int64_t adc_min   = (is_signed) \
                                     ? -((int64_t)1 << ((total_bits) - 1)) \
                                     : 0LL; \
    static const int64_t adc_max   = (is_signed) \
                                     ? (((int64_t)1 << ((total_bits) - 1)) - 1LL) \
                                     : (((int64_t)1 << (total_bits)) - 1LL); \
    *out = calc_next_datum_adc_ ## id(data, adc_vneg, adc_vpos, adc_lsb, adc_min, adc_max); \
    return true; \
}
#endif


#ifndef DEF_ADC_QUANT_PROTO
#define DEF_ADC_QUANT_PROTO(id, output_type) \
bool calc_adc_ ## id(float data, output_type *out);
#endif


#endif /* ADC_TEMPLATE_H */
