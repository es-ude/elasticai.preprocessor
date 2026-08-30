#ifndef THRESHOLDING_MAVG_POW2_TEMPLATE_H
#define THRESHOLDING_MAVG_POW2_TEMPLATE_H
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    uint16_t tap_start;
    uint16_t tap_length;
    void   *taps;
} MavgPow2Window;

#ifndef DEF_CALC_MAVG_POW2_THR
#define DEF_CALC_MAVG_POW2_THR(id, input_type) \
int32_t calc_next_datum_moving_window_average_pow2_ ##id(input_type data, MavgPow2Window *window) { \
    uint16_t win_tap_start = window->tap_start; \
    uint16_t win_tap_length = window->tap_length; \
 \
    input_type *win_tap = (input_type *) window->taps; \
    win_tap[win_tap_start] = data; \
 \
    int32_t sum = 0; \
    for (uint16_t i = 0; i < win_tap_length; i++) { \
        sum += (int32_t)win_tap[i]; \
    } \
    win_tap_start++; \
    if (win_tap_start >= win_tap_length) win_tap_start = 0; \
    window->tap_start = win_tap_start; \
    return sum; \
}
#endif//DEF_CALC_MAVG_POW2_THR

#ifndef DEF_NEW_MAVG_POW2_WINDOW_IMPL
#define DEF_NEW_MAVG_POW2_WINDOW_IMPL(id, input_type, size, logsize) \
bool calc_thresholding_mavg_pow2_ ## id(input_type data, input_type *out) { \
    static input_type window_taps[size] = {0}; \
    static const input_type div = logsize; \
    static MavgPow2Window win_params = { \
        .tap_start  = 0, \
        .tap_length = size, \
        .taps       = window_taps \
    }; \
    int32_t sum = calc_next_datum_moving_window_average_pow2_ ##id(data, &(win_params)); \
    *out = (input_type)(sum >> div); \
    return true; \
}
#endif//DEF_NEW_MAVG_POW2_WINDOW_IMPL

#ifndef DEF_NEW_MAVG_POW2_WINDOW_PROTO
#define DEF_NEW_MAVG_POW2_WINDOW_PROTO(id, input_type) \
bool calc_thresholding_mavg_pow2_ ## id(input_type data, input_type *out);
#endif//DEF_NEW_MAVG_POW2_WINDOW_PROTO

#endif//THRESHOLDING_MAVG_POW2_TEMPLATE_H
