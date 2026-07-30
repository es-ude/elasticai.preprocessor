#ifndef DOWNSAMPLING_POLY_TEMPLATE_H
#define DOWNSAMPLING_POLY_TEMPLATE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// tap-buffer for order1
typedef struct {
    size_t  tap_pos;
    size_t  tap_len;
    void   *taps;
} DoPolyTaps;

// tap-buffer for order2
typedef struct {
    size_t  tap_pos;
    size_t  tap_len;
    void   *taps;
    void   *le_state;
    void   *lep_state;
} DoPolyTwoTaps;

#ifndef DEF_DOWNSAMPLING_POLY_ONE
#define DEF_DOWNSAMPLING_POLY_ONE(id, input_type) \
bool calc_next_datum_do_poly_one_ ## id(input_type data, DoPolyTaps *s) { \
    input_type *taps = (input_type *) s->taps; \
    taps[s->tap_pos++] = data; \
    if (s->tap_pos < s->tap_len) { return false; } \
    s->tap_pos = 0; \
    input_type sum = 0; \
    for (size_t i = 0; i < s->tap_len; i++) { \
        sum += taps[i]; \
    } \
    taps[s->tap_len - 1] = sum; \
    return true; \
}
#endif//DEF_DOWNSAMPLING_POLY_ONE

#ifndef DEF_DOWNSAMPLING_POLY_ONE_IMPL
#define DEF_DOWNSAMPLING_POLY_ONE_IMPL(id, input_type, dsr) \
static DEF_DOWNSAMPLING_POLY_ONE(id, input_type) \
bool calc_do_poly_one_ ## id(input_type data, input_type *out) { \
    static input_type do_taps[dsr] = {0}; \
    static DoPolyTaps s = { \
        .tap_pos = 0, \
        .tap_len = (size_t)(dsr), \
        .taps    = do_taps, \
    }; \
    if (calc_next_datum_do_poly_one_ ## id(data, &s)) { \
        *out = ((input_type *) s.taps)[(dsr) - 1]; \
        return true; \
    } \
    return false; \
}
#endif//DEF_DOWNSAMPLING_POLY_ONE_IMPL

#ifndef DEF_DOWNSAMPLING_POLY_TWO
#define DEF_DOWNSAMPLING_POLY_TWO(id, input_type) \
bool calc_next_datum_do_poly_two_ ## id(input_type data, DoPolyTwoTaps *s) { \
    input_type *taps   = (input_type *) s->taps; \
    input_type *le_st  = (input_type *) s->le_state; \
    input_type *lep_st = (input_type *) s->lep_state; \
    taps[s->tap_pos++] = data; \
    if (s->tap_pos < s->tap_len) { return false; } \
    s->tap_pos = 0; \
    size_t n    = s->tap_len; \
    size_t pass = 0; \
    while (n > 1) { \
        input_type le  = le_st[pass]; \
        input_type lep = lep_st[pass]; \
        size_t out_pos = 0; \
        for (size_t i = 0; i < n; i++) { \
            if (i % 2 == 0) { \
                lep = le; \
                le  = taps[i]; \
            } else { \
                taps[out_pos++] = (input_type)(taps[i] + (input_type)2 * le + lep); \
            } \
        } \
        le_st[pass]  = le; \
        lep_st[pass] = lep; \
        n >>= 1; \
        pass++; \
    } \
    return true; \
}
#endif//DEF_DOWNSAMPLING_POLY_TWO

#ifndef DEF_DOWNSAMPLING_POLY_TWO_IMPL
#define DEF_DOWNSAMPLING_POLY_TWO_IMPL(id, input_type, dsr) \
static DEF_DOWNSAMPLING_POLY_TWO(id, input_type) \
bool calc_do_poly_two_ ## id(input_type data, input_type *out) { \
    static input_type do_taps[dsr]     = {0}; \
    static input_type do_le_state[dsr] = {0}; \
    static input_type do_lep_state[dsr] = {0}; \
    static DoPolyTwoTaps s = { \
        .tap_pos   = 0, \
        .tap_len   = (size_t)(dsr), \
        .taps      = do_taps, \
        .le_state  = do_le_state, \
        .lep_state = do_lep_state, \
    }; \
    if (calc_next_datum_do_poly_two_ ## id(data, &s)) { \
        *out = ((input_type *) s.taps)[0]; \
        return true; \
    } \
    return false; \
}
#endif//DEF_DOWNSAMPLING_POLY_TWO_IMPL

#ifndef DEF_DOWNSAMPLING_POLY_ONE_PROTO
#define DEF_DOWNSAMPLING_POLY_ONE_PROTO(id, input_type) \
bool calc_do_poly_one_ ## id(input_type data, input_type *out); 
#endif//DEF_DOWNSAMPLING_POLY_ONE_PROTO

#ifndef DEF_DOWNSAMPLING_POLY_TWO_PROTO
#define DEF_DOWNSAMPLING_POLY_TWO_PROTO(id, input_type) \
bool calc_do_poly_two_ ## id(input_type data, input_type *out);
#endif//DEF_DOWNSAMPLING_POLY_TWO_PROTO

#endif//DOWNSAMPLING_POLY_TEMPLATE_H
