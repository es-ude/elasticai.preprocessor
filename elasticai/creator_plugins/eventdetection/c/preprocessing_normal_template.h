#ifndef PREPROCESSING_NORMAL_TEMPLATE_H
#define PREPROCESSING_NORMAL_TEMPLATE_H

#include <stdbool.h>
#include <stdint.h>

#ifndef DEF_NEW_PREPROCESSING_NORMMAL_IMPL
    #define DEF_NEW_PREPROCESSING_NORMMAL_IMPL(id, input_type) \
        bool calc_preprocessing_normal_ ## id(input_type data, input_type *out) { \
            *out = (input_type)(data); \
            return true; \
        }
#endif

#ifndef DEF_NEW_PREPROCESSING_NORMAL_PROTO
    #define DEF_NEW_PREPROCESSING_NORMAL_PROTO(id, input_type) \
        bool calc_preprocessing_normal_ ## id(input_tpye data, input_type *out);
#endif

#endif//PREPROCESSING_NORMAL_TEMPLATE_H
