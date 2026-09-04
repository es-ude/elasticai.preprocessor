#ifndef SDA_NORMAL_CONST_TEMPLATE_H
#define SDA_NORMAL_CONST_TEMPLATE_H

#include <stdbool.h>
#include <stdint.h>

#ifndef DEF_NEW_SDA_NORMAL_CONST_IMPL
#define DEF_NEW_SDA_NORMAL_CONST_IMPL(id, input_type, threshold) \
bool calc_sda_normal_const_ ## id(input_type data, input_type *out) { \
    static const input_type thr = (input_type)(threshold); \
    *out = data; \
    return data >= thr; \
}
#endif

#ifndef DEF_NEW_SDA_NORMAL_CONST_PROTO
#define DEF_NEW_SDA_NORMAL_CONST_PROTO(id, input_type) \
bool calc_sda_normal_const_ ## id(input_type data, input_type *out);
#endif

#endif // SDA_NORMAL_CONST_TEMPLATE_H
