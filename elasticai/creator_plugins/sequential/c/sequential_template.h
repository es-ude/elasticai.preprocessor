#ifndef SEQUENTIAL_TEMPLATE_H
#define SEQUENTIAL_TEMPLATE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#ifndef SEQUENTIAL_PIPELINE
#define SEQUENTIAL_PIPELINE(in, out, num_in, num_out) \
    if ((size_t)(num_in) != (size_t)(num_out)) \
        return false; \
    for (size_t i = 0; i < (size_t)(num_in); i++) { \
        (out)[i] = (in)[i]; \
    }
#endif


#ifndef DEF_SEQUENTIAL_IMPL
#define DEF_SEQUENTIAL_IMPL(id, in_type, out_type, num_in, num_out) \
const size_t sequential_ ## id ## _n_in  = (size_t)(num_in); \
const size_t sequential_ ## id ## _n_out = (size_t)(num_out); \
bool sequential_ ## id(in_type *in, out_type *out) { \
    if (in == NULL || out == NULL) return false; \
        SEQUENTIAL_PIPELINE(in, out, num_in, num_out) \
    return true; \
}
#endif


#ifndef DEF_SEQUENTIAL_PROTO
#define DEF_SEQUENTIAL_PROTO(id, in_type, out_type) \
extern const size_t sequential_ ## id ## _n_in; \
extern const size_t sequential_ ## id ## _n_out; \
bool sequential_ ## id(in_type *in, out_type *out);
#endif


#endif /* SEQUENTIAL_TEMPLATE_H */