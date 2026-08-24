#ifndef FILTER_FIR_ALL_TEMPLATE_H
#define FILTER_FIR_ALL_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>


typedef struct {
    uint16_t tap_start;
    uint16_t tap_length;
    void *taps;
} FirAllFilter;


#ifndef DEF_CALC_FIR_DELAY
#define DEF_CALC_FIR_DELAY(id, input_type) \
input_type calc_next_datum_filter_fir_all_ ## id (input_type data, FirAllFilter *filter) { \
    input_type *filter_tap = (input_type *) filter->taps; \
    input_type value_out = filter_tap[filter->tap_start]; \
    filter_tap[filter->tap_start] = data; \
\
    if(filter->tap_start >= filter->tap_length-2){ \
        filter->tap_start = 0; \
    } else { \
        filter->tap_start++; \
    } \
    return value_out; \
}
#endif


#ifndef DEF_NEW_FIR_DELAY_IMPL
#define DEF_NEW_FIR_DELAY_IMPL(id, input_type, order) \
    static DEF_CALC_FIR_DELAY(id, input_type) \
    bool filt_fir_delay_ ## id (input_type data, input_type *out) { \
        static input_type filter_taps [order] = {0}; \
        static FirAllFilter settings = { \
            .tap_start = 0, \
            .tap_length = order, \
            .taps = filter_taps \
        }; \
        *out = calc_next_datum_filter_fir_all_ ## id (data, & (settings)); \
        return true; \
    }
#endif


#ifndef DEF_NEW_FIR_DELAY_PROTO
#define DEF_NEW_FIR_DELAY_PROTO(id, input_type) \
    bool filt_fir_delay_ ## id (input_type data, input_type *out);
#endif


#endif
