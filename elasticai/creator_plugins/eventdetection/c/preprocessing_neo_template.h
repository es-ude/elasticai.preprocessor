#ifndef PREPROCESSING_NEO_TEMPLATE_H
#define PREPROCESSING_NEO_TEMPLATE_H

#include <stdbool.h>
#include <stdint.h>

#ifndef DEF_NEW_PREPROCESSING_NEO_IMPL
    #define DEF_NEW_PREPROCESSING_NEO_IMPL(id, input_type) \
        bool calc_neo_ ## id(input_type data, input_type *out) { \
            static input_type taps[3] = {0}; \
            static int8_t count = 0; \
            if (count < 3) { \
                taps[0] = taps[1]; \
                taps[1] = taps[2]; \
                taps[2] = data; \
                count++; \
                return false; \
            } \
            taps[0] = taps[1]; \
            taps[1] = taps[2]; \
            taps[2] = data; \
            *out = taps[1] * taps[1] - taps[0] * taps[2]; \
            return true; \
        }
#endif

#ifndef DEF_NEW_PREPROCESSING_NEO_PROTO
    #define DEF_NEW_PREPROCESSING_NEO_PROTO(id, input_type) \
        bool calc_neo_ ## id(input_type data, input_type *out); 
#endif

#endif//PREPROCESSING_NEO_TEMPLATE_H
