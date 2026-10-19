#ifndef THRESHOLDING_MAVG_POW2_ABS_TEMPLATE_H
#define THRESHOLDING_MAVG_POW2_ABS_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    uint16_t tap_start;
    uint16_t tap_length;
    void    *taps;
    int32_t *tap_sum;
} MavgPow2AbsWindow;

#ifndef DEF_CALC_MAVG_POW2_ABS_THR
#define DEF_CALC_MAVG_POW2_ABS_THR(id, input_type) \
void calc_next_datum_moving_window_average_pow2_abs_ ##id(input_type data, MavgPow2AbsWindow *window) { \
    input_type *win_tap = (input_type *) window->taps; \
    int32_t *sum = window->tap_sum; \
    \
    *sum -= win_tap[window->tap_start]; \
    win_tap[window->tap_start] = (data < 0) ? -data : data; \
    *sum += win_tap[window->tap_start]; \
    if (++window->tap_start >= window->tap_length) window->tap_start = 0; \
}
#endif//DEF_CALC_MAVG_POW2_ABS_THR

#ifndef DEF_NEW_MAVG_POW2_ABS_WINDOW_IMPL
#define DEF_NEW_MAVG_POW2_ABS_WINDOW_IMPL(id, input_type, size, logsize) \
bool calc_thresholding_mavg_pow2_abs_ ## id(input_type data, input_type *out) { \
    static input_type window_taps[size] = {0}; \
    static const input_type div = logsize; \
    static int32_t sum = 0; \
    static MavgPow2AbsWindow win_params = { \
        .tap_start  = 0, \
        .tap_length = size, \
        .taps       = window_taps, \
        .tap_sum    = &sum \
    }; \
    calc_next_datum_moving_window_average_pow2_abs_ ##id(data, &(win_params)); \
    *out = (input_type)(*win_params.tap_sum >> div); \
    return true; \
}
#endif//DEF_NEW_MAVG_POW2_ABS_WINDOW_IMPL

#ifndef DEF_NEW_MAVG_POW2_ABS_WINDOW_PROTO
#define DEF_NEW_MAVG_POW2_ABS_WINDOW_PROTO(id, input_type) \
bool calc_thresholding_mavg_pow2_abs_ ## id(input_type data, input_type *out);
#endif//DEF_NEW_MAVG_POW2_ABS_WINDOW_PROTO

#endif//THRESHOLDING_MAVG_POW2_ABS_TEMPLATE_H
