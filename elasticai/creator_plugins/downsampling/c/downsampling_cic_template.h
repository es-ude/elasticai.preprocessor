#ifndef DOWNSAMPLING_CIC_TEMPLATE_H
#define DOWNSAMPLING_CIC_TEMPLATE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int64_t *integrator_yn; // accumulator state
    int64_t *comb_xn;       // comb filter state
    int64_t *comb_xnm;      // comb filter state memory
    size_t   count;
    size_t   dsr;
    int      num_stages;
    int64_t  gain;
} CicTaps;


#ifndef DOWNSAMPLING_CIC_OUTPUT_LENGTH
#define DOWNSAMPLING_CIC_OUTPUT_LENGTH(id, dsr) \
size_t get_downsampling_cic_output_length_ ## id(size_t input_length) { \
    return input_length / (size_t)(dsr); \
}
#endif // DOWNSAMPLING_CIC_OUTPUT_LENGTH

#ifndef DEF_DOWNSAMPLING_CIC
#define DEF_DOWNSAMPLING_CIC(id, input_type) \
bool calc_next_datum_cic_ ## id(input_type data, input_type *out, CicTaps *s) { \
    int64_t z = (int64_t)data; \
    for (int i = 0; i < s->num_stages; i++) { \
        s->integrator_yn[i] += z; \
        z = s->integrator_yn[i]; \
    } \
    if (s->count % s->dsr == 0) { \
        for (int i = 0; i < s->num_stages; i++) { \
            s->comb_xnm[i] = s->comb_xn[i]; \
            s->comb_xn[i]  = z; \
            z = s->comb_xn[i] - s->comb_xnm[i]; \
        } \
        *out = (input_type)(z / s->gain); \
        s->count++; \
        return true; \
    } \
    s->count++; \
    return false; \
}
#endif // DEF_DOWNSAMPLING_CIC


#ifndef DEF_DOWNSAMPLING_CIC_IMPL
#define DEF_DOWNSAMPLING_CIC_IMPL(id, input_type, cic_stages, cic_dsr) \
static DEF_DOWNSAMPLING_CIC(id, input_type) \
bool calc_cic_ ## id(input_type data, input_type *out) { \
    static int64_t cic_int_yn_  ## id[cic_stages] = {0}; \
    static int64_t cic_cmb_xn_  ## id[cic_stages] = {0}; \
    static int64_t cic_cmb_xnm_ ## id[cic_stages] = {0}; \
    static CicTaps cic_s_       ## id = { \
        .integrator_yn  = cic_int_yn_  ## id, \
        .comb_xn        = cic_cmb_xn_  ## id, \
        .comb_xnm       = cic_cmb_xnm_ ## id, \
        .dsr        = (size_t)(cic_dsr), \
        .num_stages = (cic_stages), \
        .gain       = 0, \
    }; \
    if (cic_s_ ## id.gain == 0) { \
        cic_s_ ## id.gain = 1; \
        for (int i = 0; i < (cic_stages); i++) cic_s_ ## id.gain *= (int64_t)(cic_dsr); \
    } \
    return calc_next_datum_cic_ ## id(data, out, &(cic_s_ ## id)); \
}
#endif // DEF_DOWNSAMPLING_CIC_IMPL


#ifndef DEF_DOWNSAMPLING_CIC_PROTO
#define DEF_DOWNSAMPLING_CIC_PROTO(id, input_type) \
size_t get_downsampling_cic_output_length_ ## id(size_t input_length); \
bool calc_cic_ ## id(input_type data, input_type *out);
#endif // DEF_DOWNSAMPLING_CIC_PROTO


#endif // DOWNSAMPLING_CIC_TEMPLATE_H
