#ifndef PIPELINE_TEMPLATE_H
#define PIPELINE_TEMPLATE_H

#include <stdbool.h>
#include <stdint.h>

#include "filter_iir_template.h"
#include "downsampling_poly_template.h"
#include "windower_template.h"


/*
 * DEF_PIPELINE_IMPL — instantiates a filter → downsampling → windower pipeline.
 *
 * Parameters:
 *   id           unique identifier, appended to all generated function names
 *   input_type   C integer type used throughout (e.g. int32_t, int16_t, int8_t)
 *   coeff_lgth   number of IIR filter coefficients (one direction, i.e. order + 1)
 *   tap_lgth     length of the IIR tap-delay buffer (= filter order)
 *   dsr          downsampling ratio (decimation factor)
 *   wl           window length in samples (after downsampling)
 *   nshift       number of samples to shift between consecutive windows
 *   ...          IIR filter coefficients (b0,b1,...,bN, a1,...,aN)
 *
 * Generated function:
 *   bool calc_pipeline_{id}(input_type data, input_type *out)
 *
 *   Call once per input sample.
 *   Returns true when out[0..wl-1] contains a complete new window.
 *   Returns false while the pipeline is still filling up.
 *
 * Example instantiation (bandpass 100–300 Hz, DSR=8, window=100 samples):
 *   DEF_PIPELINE_IMPL(0, int32_t, 3, 2, 8, 100, 100,
 *       1.0, -2.0, 1.0,   // b coefficients
 *       -1.8, 0.82        // a coefficients
 *   )
 *
 * Calling convention:
 *   int32_t window[100];
 *   if (calc_pipeline_0(new_sample, window)) {
 *       // window[0..99] holds the new frame — process it
 *   }
 */
#ifndef DEF_PIPELINE_IMPL
#define DEF_PIPELINE_IMPL(id, input_type, coeff_lgth, tap_lgth, dsr, wl, nshift, ...) \
    DEF_NEW_IIR_FILTER_IMPL(id, input_type, coeff_lgth, tap_lgth, __VA_ARGS__) \
    DEF_DOWNSAMPLING_POLY_ONE_IMPL(id, input_type, dsr) \
    DEF_WINDOWER_IMPL(id, input_type, wl, nshift) \
    bool calc_pipeline_ ## id(input_type data, input_type *out) { \
        input_type filt_out; \
        if (!filt_iir_ ## id(data, &filt_out)) { return false; } \
        input_type down_out; \
        if (!calc_do_poly_one_ ## id(filt_out, &down_out)) { return false; } \
        return calc_windower_ ## id(down_out, out); \
    }
#endif /* DEF_PIPELINE_IMPL */


#ifndef DEF_PIPELINE_PROTO
#define DEF_PIPELINE_PROTO(id, input_type) \
    bool calc_pipeline_ ## id(input_type data, input_type *out);
#endif /* DEF_PIPELINE_PROTO */


#endif /* PIPELINE_TEMPLATE_H */
