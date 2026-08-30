#ifndef THRESHOLDING_MAVG_ABS_TEMPLATE_H
#define THRESHOLDING_MAVG_ABS_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    uint16_t tap_start;
    uint16_t tap_length;
    void   *taps;
} MavgAbsWindow;

#ifndef DEF_CALC_MAVG_ABS_THR
#define DEF_CALC_MAVG_ABS_THR(id, input_type) \
float calc_next_datum_moving_window_average_abs_ ##id(input_type data, MavgAbsWindow *window) { \
    uint16_t win_tap_start = window->tap_start; \
    uint16_t win_tap_length = window->tap_length; \
 \
    input_type *win_tap = (input_type *) window->taps; \
    win_tap[win_tap_start] = (data < 0) ? -data : data; \
 \
    int32_t sum = 0; \
    for (uint16_t i = 0; i < win_tap_length; i++) { \
        sum += (int32_t)win_tap[i]; \
    } \
    win_tap_start++; \
    if (win_tap_start >= win_tap_length) win_tap_start = 0; \
    window->tap_start = win_tap_start; \
    return (float)sum / (float)win_tap_length; \
}
#endif//DEF_CALC_MAVG_ABS_THR

#ifndef DEF_NEW_MAVG_ABS_WINDOW_IMPL
#define DEF_NEW_MAVG_ABS_WINDOW_IMPL(id, input_type, size) \
bool calc_thresholding_mavg_abs_ ## id(input_type data, input_type *out) { \
    static input_type window_taps[size] = {0}; \
    static MavgAbsWindow win_params = { \
        .tap_start = 0, \
        .tap_length = size, \
        .taps = window_taps \
    }; \
    *out = (input_type)floorf(calc_next_datum_moving_window_average_abs_ ##id(data, &(win_params))); \
    return true; \
}
#endif//DEF_NEW_MAVG_ABS_WINDOW_IMPL

#ifndef DEF_NEW_MAVG_ABS_WINDOW_PROTO
#define DEF_NEW_MAVG_ABS_WINDOW_PROTO(id, input_type) \
bool calc_thresholding_mavg_abs_ ## id(input_type data, input_type *out);
#endif//DEF_NEW_MAVG_ABS_WINDOW_PROTO

#endif//THRESHOLDING_MAVG_ABS_TEMPLATE_H
