#ifndef THRESHOLDING_MAVG_TEMPLATE_H
#define THRESHOLDING_MAVG_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    uint16_t tap_start;
    uint16_t tap_length;
    void    *taps;
    int32_t *tap_sum;
} MavgWindow;

#ifndef DEF_CALC_MAVG_THR
#define DEF_CALC_MAVG_THR(id, input_type) \
float calc_next_datum_moving_window_average_ ##id(input_type data, MavgWindow *window) { \
    input_type *win_tap = (input_type *) window->taps; \
    int32_t *sum = window->tap_sum; \
    \
    *sum += data - win_tap[window->tap_start]; \
    win_tap[window->tap_start] = data; \
    if (++window->tap_start >= window->tap_length) window->tap_start = 0; \
    \
    return (float)*sum / (float)window->tap_length; \
}
#endif//DEF_CALC_MAVG_THR

#ifndef DEF_NEW_MAVG_WINDOW_IMPL
#define DEF_NEW_MAVG_WINDOW_IMPL(id, input_type, size) \
bool calc_thresholding_mavg_ ## id(input_type data, input_type *out) { \
    static input_type window_taps[size] = {0}; \
    static int32_t sum = 0; \
    static MavgWindow win_params = { \
        .tap_start = 0, \
        .tap_length = size, \
        .taps = window_taps, \
        .tap_sum = &sum \
    }; \
    *out = (input_type)floorf(calc_next_datum_moving_window_average_ ##id(data, &(win_params))); \
    return true; \
}
#endif//DEF_NEW_MAVG_WINDOW_IMPL

#ifndef DEF_NEW_MAVG_WINDOW_PROTO
#define DEF_NEW_MAVG_WINDOW_PROTO(id, input_type) \
bool calc_thresholding_mavg_ ## id(input_type data, input_type *out);
#endif//DEF_NEW_MAVG_WINDOW_PROTO

#endif//THRESHOLDING_MAVG_TEMPLATE_H
