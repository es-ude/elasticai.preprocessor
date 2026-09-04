#ifndef SDA_ABSOLUTE_CONST_TEMPLATE_H
#define SDA_ABSOLUTE_CONST_TEMPLATE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef DEF_NEW_SDA_ABSOLUTE_CONST_IMPL
#define DEF_NEW_SDA_ABSOLUTE_CONST_IMPL(id, input_type, threshold) \
bool calc_sda_abs_const_ ## id(input_type data, input_type *out) { \
    static const input_type thr = (input_type)(threshold); \
    *out = (input_type)abs(data); \
    return (input_type)abs(data) >= thr; \
}
#endif

#ifndef DEF_NEW_SDA_ABSOLUTE_CONST_PROTO
#define DEF_NEW_SDA_ABSOLUTE_CONST_PROTO(id, input_type) \
bool calc_sda_abs_const_ ## id(input_type data, input_type *out);
#endif

#endif // SDA_ABSOLUTE_CONST_TEMPLATE_H
