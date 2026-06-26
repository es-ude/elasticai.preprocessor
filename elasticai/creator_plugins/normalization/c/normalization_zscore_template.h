#ifndef NORMALIZATION_ZSCORE_TEMPLATE_H
#define NORMALIZATION_ZSCORE_TEMPLATE_H

#include <math.h>
#include <stdint.h>


#ifndef DEF_NEW_NORMALIZATION_ZSCORE_IMPL
#define DEF_NEW_NORMALIZATION_ZSCORE_IMPL(id, input_type) \
void normalize_zscore_ ## id( \
    const input_type *input, float *output, uint32_t length \
) { \
    if (length == 0) { \
        return; \
    } \
    float mean = 0.0f; \
    for (uint32_t index = 0; index < length; ++index) { \
        mean += (float)input[index]; \
    } \
    mean /= (float)length; \
    float variance = 0.0f; \
    for (uint32_t index = 0; index < length; ++index) { \
        const float value = (float)input[index] - mean; \
        variance += value * value; \
    } \
    const float std = sqrtf(variance / (float)length); \
    for (uint32_t index = 0; index < length; ++index) { \
        output[index] = std == 0.0f ? NAN : ((float)input[index] - mean) / std; \
    } \
}
#endif


#ifndef DEF_NEW_NORMALIZATION_ZSCORE_PROTO
#define DEF_NEW_NORMALIZATION_ZSCORE_PROTO(id, input_type) \
void normalize_zscore_ ## id( \
    const input_type *input, float *output, uint32_t length \
);
#endif


#endif
