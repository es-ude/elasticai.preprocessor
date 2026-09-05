#ifndef WINDOWER_SEQUENCE_TEMPLATE_H
#define WINDOWER_SEQUENCE_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>


typedef struct {
    uint16_t window_length;
    uint16_t count;
    void    *buf;
} WindowerSequenceTaps;


#ifndef DEF_WINDOWER_SEQUENCE_
#define DEF_WINDOWER_SEQUENCE_(id, input_type) \
bool calc_next_datum_windower_sequence_ ## id(input_type data, WindowerTaps *taps) { \
    input_type *buf = (input_type *) taps->buf; \
    uint16_t wl = taps->window_length; \
    for (uint16_t i = 1; i < wl; i++) buf[i - 1] = buf[i]; \
    buf[wl - 1] = data; \
    taps->count++; \
    if (taps->count >= wl) { \
        taps->count = 0; \
        return true; \
    } \
    return false; \
}
#endif


#ifndef DEF_WINDOWER_SEQUENCE_IMPL
#define DEF_WINDOWER_SEQUENCE_IMPL(id, input_type, wl) \
static DEF_WINDOWER_SEQUENCE(id, input_type) \
bool calc_windower_sequence_ ## id(input_type data, input_type *out) { \
    static input_type windower_buf[wl]; \
    static WindowerSequenceTaps taps = { \
        .window_length = (wl), \
        .count         = 0, \
        .buf           = windower_buf, \
    }; \
    if (calc_next_datum_windower_sequence_ ## id(data, &taps)) { \
        input_type *buf = (input_type *) taps.buf; \
        for (uint16_t i = 0; i < (uint16_t)(wl); i++) { \
            out[i] = buf[i]; \
        } \
        return true; \
    } \
    return false; \
}
#endif


#ifndef DEF_WINDOWER_SEQUENCE_PROTO
#define DEF_WINDOWER_SEQUENCE_PROTO(id, input_type) \
bool calc_windower_ ## id(input_type data, input_type *out);
#endif


#endif /* WINDOWER_SEQUENCE_TEMPLATE_H */
