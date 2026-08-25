#ifndef EVENTDETECTION_TEMPLATE_H
#define EVENTDETECTION_TEMPLATE_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
        NORMAL,
        POS_HYST,
        NEG_HYST,
        DOUBLE_HYST,
    } EventDetectionHysteresisType;

#ifndef EVENT_SETTINGS
#define EVENT_SETTINGS(id, input_type) \
typedef struct { \
    input_type hysteresis; \
    bool       int_state; \
    bool       out_invert; \
    EventDetectionHysteresisType hysteresis_type; \
} EventSettings_ ## id;
#endif

#ifndef TYPE_HYSTERESIS
#define TYPE_HYSTERESIS(id, input_type) \
void get_limits_ ## id(EventSettings_ ## id settings, input_type threshold, input_type *limits) { \
    input_type thr_zero = threshold; \
    input_type thr_pos = thr_zero + settings.hysteresis; \
    input_type thr_neg = thr_zero - settings.hysteresis; \
    switch (settings.hysteresis_type) { \
        case NORMAL: \
            limits[0] = thr_zero; \
            limits[1] = thr_zero; \
            break; \
        case POS_HYST: \
            limits[0] = thr_pos; \
            limits[1] = thr_zero; \
            break; \
        case NEG_HYST: \
            limits[0] = thr_zero; \
            limits[1] = thr_neg; \
            break; \
        case DOUBLE_HYST: \
            limits[0] = thr_pos; \
            limits[1] = thr_neg; \
            break; \
    } \
}
#endif

#ifndef DEF_NEW_EVENTDETECTION_IMPL
#define DEF_NEW_EVENTDETECTION_IMPL(id, input_type, hyster, hyster_type, invert) \
bool is_event_ ## id(input_type data, input_type threshold) {  \
    static EventSettings_ ## id settings = { \
        .hysteresis      = hyster, \
        .int_state       = invert, \
        .out_invert      = invert, \
        .hysteresis_type = hyster_type, \
    }; \
    input_type limits[2] = {0}; \
    get_limits_ ## id(settings, threshold, limits); \
    bool state_return = false; \
    if (settings.int_state) { \
        if (data >= limits[1]) { \
            state_return = true; \
        } else \
            state_return = false; \
    } else { \
        if (data >= limits[0]) { \
            state_return = true; \
        } else { \
            state_return = false; \
        } \
    }; \
    settings.int_state = state_return; \
    return (settings.out_invert) ? !state_return : state_return; \
}
#endif

#ifndef DEF_NEW_EVENTDETECTION_PROTO
    #define DEF_NEW_EVENTDETECTION_PROTO(id, input_type) \
        bool is_event_ ## id(input_type data, input_type threshold);
    #endif
#endif
