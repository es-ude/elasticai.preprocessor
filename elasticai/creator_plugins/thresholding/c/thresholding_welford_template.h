#ifndef THRESHOLDING_WELFORD_TEMPLATE_H
#define THRESHOLDING_WELFORD_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>
#include <math.h>


typedef struct {
    uint32_t n;
    float   mean;
    float   sigma;
} WelfordTaps;


#ifndef DEF_CALC_WELFORD_THR
#define DEF_CALC_WELFORD_THR(id, input_type) \
void calc_next_datum_thresholding_welford_ ## id(input_type data, WelfordTaps *taps) { \
    taps->n++; \
    float mean_old  = taps->mean; \
    taps->mean      += ((float)data - taps->mean)  / (float)taps->n; \
    taps->sigma     += (((float)data - taps->mean) * ((float)data - mean_old) \
                        - taps->sigma) / (float)taps->n; \
}
#endif


#ifndef DEF_WELFORD_THR_IMPL
#define DEF_WELFORD_THR_IMPL(id, input_type, gain_val) \
static DEF_CALC_WELFORD_THR(id, input_type) \
bool calc_thresholding_welford_ ## id(input_type data, input_type *out) { \
    static WelfordTaps taps = { \
        .n = 0,  \
        .mean = 0.0f,  \
        .sigma = 0.0f}; \
    static const float thr_gain = (float)(gain_val); \
    static input_type stable_out = 0; \
    if (taps.n < UINT32_MAX) { \
        calc_next_datum_thresholding_welford_ ## id(data, &taps); \
        if (taps.n < 2) return false; \
        stable_out = (input_type)(thr_gain * sqrtf((float)taps.sigma)); \
    } \
    *out = stable_out; \
    return true; \
}
#endif


#ifndef DEF_WELFORD_THR_PROTO
#define DEF_WELFORD_THR_PROTO(id, input_type) \
bool calc_thresholding_welford_ ## id(input_type data, input_type *out);
#endif


#endif /* THRESHOLDING_WELFORD_TEMPLATE_H */
