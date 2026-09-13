#ifndef DOWNSAMPLING_FLOAT32_TEMPLATE_H
#define DOWNSAMPLING_FLOAT32_TEMPLATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#ifndef DEF_DOWNSAMPLING_FLOAT32_PROTO
#define DEF_DOWNSAMPLING_FLOAT32_PROTO(id) \
uint32_t get_downsampling_float32_ ## id ## _output_length( \
    uint32_t input_length \
); \
bool downsample_float32_ ## id( \
    const float *input, \
    uint32_t input_length, \
    float *output, \
    uint32_t output_capacity \
);
#endif


#ifndef DEF_DOWNSAMPLING_FLOAT32_IMPL
#define DEF_DOWNSAMPLING_FLOAT32_IMPL(id, factor) \
static const uint32_t downsampling_float32_factor_ ## id = factor; \
uint32_t get_downsampling_float32_ ## id ## _output_length( \
    uint32_t input_length \
) { \
    if (input_length == 0u) { \
        return 0u; \
    } \
    return 1u + (input_length - 1u) \
        / downsampling_float32_factor_ ## id; \
} \
bool downsample_float32_ ## id( \
    const float *input, \
    uint32_t input_length, \
    float *output, \
    uint32_t output_capacity \
) { \
    const uint32_t output_length = \
        get_downsampling_float32_ ## id ## _output_length(input_length); \
    if (output_length == 0u) { \
        return true; \
    } \
    if (input == NULL || output == NULL || output_capacity < output_length) { \
        return false; \
    } \
    for (uint32_t output_index = 0u; output_index < output_length; ++output_index) { \
        output[output_index] = input[ \
            output_index * downsampling_float32_factor_ ## id \
        ]; \
    } \
    return true; \
}
#endif


#endif
