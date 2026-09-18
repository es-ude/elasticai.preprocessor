#ifndef FILTER_IIR_FLOAT32_TEMPLATE_H
#define FILTER_IIR_FLOAT32_TEMPLATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#ifndef DEF_NEW_IIR_FLOAT32_FILTER_PROTO
#define DEF_NEW_IIR_FLOAT32_FILTER_PROTO(id, tap_lgth) \
    typedef struct { \
        uint16_t tap_start; \
        float taps[tap_lgth]; \
    } IirFilterFloat32State_ ## id; \
    void reset_filter_iir_float32_ ## id(IirFilterFloat32State_ ## id *state); \
    bool filt_iir_float32_ ## id( \
        IirFilterFloat32State_ ## id *state, float data, float *out \
    );
#endif


#ifndef DEF_NEW_IIR_FLOAT32_FILTER_IMPL
#define DEF_NEW_IIR_FLOAT32_FILTER_IMPL(id, coeff_lgth, tap_lgth, ...) \
    static const float filter_iir_float32_coefficients_ ## id[] = {__VA_ARGS__}; \
    void reset_filter_iir_float32_ ## id(IirFilterFloat32State_ ## id *state) { \
        if (state == NULL) { \
            return; \
        } \
        state->tap_start = 0u; \
        for (uint16_t tap = 0u; tap < tap_lgth; ++tap) { \
            state->taps[tap] = 0.0f; \
        } \
    } \
    bool filt_iir_float32_ ## id( \
        IirFilterFloat32State_ ## id *state, float data, float *out \
    ) { \
        if (state == NULL || out == NULL) { \
            return false; \
        } \
        int16_t pos_tap = (int16_t)state->tap_start - 1; \
        if (pos_tap < 0) { \
            pos_tap = (int16_t)tap_lgth - 1; \
        } \
        float value_input = data; \
        for (uint16_t coefficient = 1u; coefficient < coeff_lgth; ++coefficient) { \
            value_input -= filter_iir_float32_coefficients_ ## id[coefficient] \
                * state->taps[pos_tap]; \
            --pos_tap; \
            if (pos_tap < 0) { \
                pos_tap = (int16_t)tap_lgth - 1; \
            } \
        } \
        pos_tap = (int16_t)state->tap_start - 1; \
        if (pos_tap < 0) { \
            pos_tap = (int16_t)tap_lgth - 1; \
        } \
        float value_output = filter_iir_float32_coefficients_ ## id[coeff_lgth] * value_input; \
        for (uint16_t coefficient = 1u; coefficient < coeff_lgth; ++coefficient) { \
            value_output += filter_iir_float32_coefficients_ ## id[coeff_lgth + coefficient] \
                * state->taps[pos_tap]; \
            --pos_tap; \
            if (pos_tap < 0) { \
                pos_tap = (int16_t)tap_lgth - 1; \
            } \
        } \
        state->taps[state->tap_start] = value_input; \
        ++state->tap_start; \
        if (state->tap_start >= tap_lgth) { \
            state->tap_start = 0u; \
        } \
        *out = value_output; \
        return true; \
    }
#endif


#endif
