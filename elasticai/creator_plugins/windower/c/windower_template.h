#ifndef WINDOWER_TEMPLATE_H
#define WINDOWER_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>


typedef struct {
    uint16_t window_length;
    uint16_t num_shift;
    uint16_t count;
    uint16_t shift_cnt;
    void    *buf;
} WindowerTaps;


#ifndef DEF_WINDOWER
#define DEF_WINDOWER(id, input_type) \
bool calc_next_datum_windower_ ## id(input_type data, WindowerTaps *taps) { \
    input_type *buf = (input_type *) taps->buf; \
    uint16_t wl = taps->window_length; \
    for (uint16_t i = 1; i < wl; i++) buf[i - 1] = buf[i]; \
    buf[wl - 1] = data; \
    if (taps->count < wl) { \
        taps->count++; \
        if (taps->count == wl) { \
            taps->shift_cnt = 0; \
            return true; \
        } \
        return false; \
    } \
    taps->shift_cnt++; \
    if (taps->shift_cnt >= taps->num_shift) { \
        taps->shift_cnt = 0; \
        return true; \
    } \
    return false; \
}
#endif


#ifndef DEF_WINDOWER_IMPL
#define DEF_WINDOWER_IMPL(id, input_type, wl, nshift) \
static DEF_WINDOWER(id, input_type) \
bool calc_windower_ ## id(input_type data, input_type *out) { \
    static input_type windower_buf[wl]; \
    static WindowerTaps taps = { \
        .window_length = (wl), \
        .num_shift     = (nshift), \
        .count         = 0, \
        .shift_cnt     = 0, \
        .buf           = windower_buf, \
    }; \
    if (calc_next_datum_windower_ ## id(data, &taps)) { \
        input_type *buf = (input_type *) taps.buf; \
        for (uint16_t i = 0; i < (uint16_t)(wl); i++) { \
            out[i] = buf[i]; \
        } \
        return true; \
    } \
    return false; \
}
#endif


#ifndef DEF_WINDOWER_PROTO
#define DEF_WINDOWER_PROTO(id, input_type) \
bool calc_windower_ ## id(input_type data, input_type *out);
#endif


#endif /* WINDOWER_TEMPLATE_H */
