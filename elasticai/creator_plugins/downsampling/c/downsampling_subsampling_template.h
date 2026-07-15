#ifndef DOWNSAMPLING_SUBSAMPLING_TEMPLATE_H
#define DOWNSAMPLING_SUBSAMPLING_TEMPLATE_H

#include <stddef.h>
#include <stdint.h>


#ifndef DEF_DOWNSAMPLING_SUBSAMPLING_OUTPUT_LENGTH
#define DEF_DOWNSAMPLING_SUBSAMPLING_OUTPUT_LENGTH(id, factor) \
size_t get_downsampling_subsampling_output_length_ ## id(size_t input_length) { \
    return (input_length + factor - 1u) / factor; \
}
#endif


#ifndef DEF_DOWNSAMPLING_SUBSAMPLING_IMPL
#define DEF_DOWNSAMPLING_SUBSAMPLING_IMPL(id, input_type, factor) \
void downsample_subsampling_ ## id( \
    const input_type *input, input_type *output, size_t input_length, uint8_t augment \
) { \
    const size_t output_length = get_downsampling_subsampling_output_length_ ## id(input_length); \
    const size_t offsets = augment ? factor : 1u; \
    for (size_t offset = 0u; offset < offsets; ++offset) { \
        for (size_t output_index = 0u; output_index < output_length; ++output_index) { \
            const size_t input_index = offset + output_index * factor; \
            output[offset * output_length + output_index] = \
                input_index < input_length ? input[input_index] : (input_type)0; \
        } \
    } \
}
#endif


#ifndef DEF_DOWNSAMPLING_SUBSAMPLING_PROTO
#define DEF_DOWNSAMPLING_SUBSAMPLING_PROTO(id, input_type) \
size_t get_downsampling_subsampling_output_length_ ## id(size_t input_length); \
void downsample_subsampling_ ## id( \
    const input_type *input, input_type *output, size_t input_length, uint8_t augment \
);
#endif


#endif
