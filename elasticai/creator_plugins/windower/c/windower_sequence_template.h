#ifndef WINDOWER_SEQUENCE_TEMPLATE_H
#define WINDOWER_SEQUENCE_TEMPLATE_H

#include <stdbool.h>
#include <stdint.h>

// Sequence windower: collects exactly wl samples, then outputs them as one window.
// Equivalent to sliding windower with overlap=0 (num_shift = window_length).

#ifndef DEF_WINDOWER_SEQUENCE_IMPL
#define DEF_WINDOWER_SEQUENCE_IMPL(id, input_type, wl) \
    bool calc_windower_sequence_ ## id(input_type data, input_type *out) { \
        static input_type buf[wl]; \
        static uint16_t idx = 0; \
        buf[idx++] = data; \
        if (idx == (uint16_t)(wl)) { \
            idx = 0; \
            for (uint16_t i = 0; i < (uint16_t)(wl); i++) out[i] = buf[i]; \
            return true; \
        } \
        return false; \
    }
#endif

#ifndef DEF_WINDOWER_SEQUENCE_PROTO
#define DEF_WINDOWER_SEQUENCE_PROTO(id, input_type) \
    bool calc_windower_sequence_ ## id(input_type data, input_type *out);
#endif

#endif /* WINDOWER_SEQUENCE_TEMPLATE_H */
