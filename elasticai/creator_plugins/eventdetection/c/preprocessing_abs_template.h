#ifndef PREPROCESSING_ABS_TEMPLATE_H
#define PREPROCESSING_ABS_TEMPLATE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef DEF_NEW_PREPROCESSING_ABS_IMPL
    #define DEF_NEW_PREPROCESSING_ABS_IMPL(id, input_type) \
        bool calc_preprocessing_abs_ ## id(input_type data, input_type *out) { \
            *out = (input_type)abs(data); \
            return true; \
        }
#endif

#ifndef DEF_NEW_PREPROCESSING_ABS_PROTO
    #define DEF_NEW_PREPROCESSING_ABS_PROTO(id, input_type) \
        bool calc_preprocessing_abs_ ## id(input_tpye data, input_type *out);
#endif

#endif//PREPROCESSING_ABS_TEMPLATE_H
