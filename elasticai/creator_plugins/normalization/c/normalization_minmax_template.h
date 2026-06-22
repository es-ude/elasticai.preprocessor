#ifndef NORMALIZATION_MINMAX_TEMPLATE_H
#define NORMALIZATION_MINMAX_TEMPLATE_H

#include <math.h>
#include <stdint.h>


#ifndef DEF_NEW_NORMALIZATION_MINMAX_IMPL
#define DEF_NEW_NORMALIZATION_MINMAX_IMPL(id, input_type) \
void normalize_minmax_ ## id( \
    const input_type *input, float *output, uint32_t length \
) { \
    float scale = 0.0f; \
    for (uint32_t index = 0; index < length; ++index) { \
        const float magnitude = fabsf((float)input[index]); \
        if (magnitude > scale) { \
            scale = magnitude; \
        } \
    } \
    for (uint32_t index = 0; index < length; ++index) { \
        output[index] = scale == 0.0f ? NAN : (float)input[index] / scale; \
    } \
}
#endif


#ifndef DEF_NEW_NORMALIZATION_MINMAX_PROTO
#define DEF_NEW_NORMALIZATION_MINMAX_PROTO(id, input_type) \
void normalize_minmax_ ## id( \
    const input_type *input, float *output, uint32_t length \
);
#endif


#endif
