#ifndef DOWNSAMPLING_SIMPLE_TEMPLATE_H
#define DOWNSAMPLING_SIMPLE_TEMPLATE_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint16_t tap_start;
    uint16_t tap_length;
    void *output;
    void *taps;
} DoSimpleTaps;

#ifndef DOWNSAMPLING_SIMPLE_OUTPUT_LENGTH
#define DOWNSAMPLING_SIMPLE_OUTPUT_LENGTH(id, factor) \
size_t get_downsampling_simple_output_length_ ## id(size_t input_length) { \
    return (input_length / (size_t)factor); \
}
#endif // DOWNSAMPLING_SIMPLE_OUTPUT_LENGTH

#ifndef DEF_DOWNSAMPLING_SIMPLE
#define DEF_DOWNSAMPLING_SIMPLE(id, input_type) \
bool calc_next_datum_do_simple_ ## id(input_type data, DoSimpleTaps *tap_memory) { \
    uint16_t do_tap_start = tap_memory->tap_start; \
    uint16_t do_tap_length = tap_memory->tap_length; \
    input_type *do_output = (input_type *) tap_memory->output; \
    input_type *do_taps = (input_type *) tap_memory->taps; \
    do_taps[do_tap_start] = data; \
    \
    int32_t sum = 0; \
    for (size_t pos_tap = 0; pos_tap < do_tap_length; pos_tap++) { \
        sum += (int32_t)do_taps[pos_tap]; \
    } \
    do_tap_start++; \
    tap_memory->tap_start = do_tap_start; \
    if(do_tap_start >= do_tap_length) { \
        tap_memory->tap_start = 0; \
        *do_output = (input_type) (sum / do_tap_length); \
        return true; \
    } \
    return false; \
}
#endif // DEF_DOWNSAMPLING_SIMPLE

#ifndef DEF_NEW_DO_SIMPLE_TAP_IMPL
#define DEF_NEW_DO_SIMPLE_TAP_IMPL(id, input_type, dsr) \
static DEF_DOWNSAMPLING_SIMPLE(id, input_type) \
bool calc_do_simple_ ## id(input_type data, input_type *out) { \
    static input_type do_taps[dsr] = {0}; \
    static input_type do_output_val = 0; \
    static DoSimpleTaps settings = { \
        .tap_length = dsr, \
        .tap_start = 0, \
        .taps = do_taps, \
        .output = &do_output_val, \
    }; \
    if (calc_next_datum_do_simple_ ## id(data, &(settings))) { \
    *out = *((input_type *) settings.output); \
    return true; \
    } \
    return false; \
}
#endif // DEF_NEW_DO_SIMPLE_TAP_IMPL

#ifndef DEF_NEW_DO_SIMPLE_PROTO
#define DEF_NEW_DO_SIMPLE_PROTO(id, input_type) \
bool calc_do_simple_ ## id(input_type data, input_type *out);
#endif // DEF_NEW_DO_SIMPLE_PROTO

#endif // DOWNSAMPLING_SIMPLE_TEMPLATE_H
