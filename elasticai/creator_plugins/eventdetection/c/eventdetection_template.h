#ifndef EVENTDETECTION_TEMPLATE_H
#define EVENTDETECTION_TEMPLATE_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
        EVENTDETECTION_NORMAL,
        EVENTDETECTION_POS_HYST,
        EVENTDETECTION_NEG_HYST,
        EVENTDETECTION_DOUBLE_HYST,
    } EventDetectionHysteresisType;

#ifndef EVENT_SETTINGS
#define EVENT_SETTINGS(id, input_type) \
typedef struct { \
    float      hysteresis; \
    bool       int_state; \
    EventDetectionHysteresisType hysteresis_type; \
} EventSettings_ ## id;
#endif//EVENT_SETTINGS

#ifndef TYPE_HYSTERESIS
#define TYPE_HYSTERESIS(id, input_type) \
void get_limits_ ## id(EventSettings_ ## id settings, input_type threshold, input_type *limits) { \
    input_type thr_zero = threshold; \
    input_type thr_pos = thr_zero + thr_zero * settings.hysteresis; \
    input_type thr_neg = thr_zero - thr_zero * settings.hysteresis; \
    switch (settings.hysteresis_type) { \
        case EVENTDETECTION_NORMAL: \
            limits[0] = thr_zero; \
            limits[1] = thr_zero; \
            break; \
        case EVENTDETECTION_POS_HYST: \
            limits[0] = thr_pos; \
            limits[1] = thr_zero; \
            break; \
        case EVENTDETECTION_NEG_HYST: \
            limits[0] = thr_zero; \
            limits[1] = thr_neg; \
            break; \
        case EVENTDETECTION_DOUBLE_HYST: \
            limits[0] = thr_pos; \
            limits[1] = thr_neg; \
            break; \
    } \
}
#endif//TYPE_HYSTERESIS

#ifndef DEF_NEW_EVENTDETECTION_IMPL
#define DEF_NEW_EVENTDETECTION_IMPL(id, input_type, hyster, hyster_type) \
bool is_event_ ## id(input_type data, input_type threshold) {  \
    static EventSettings_ ## id settings = { \
        .hysteresis      = hyster, \
        .int_state       = false, \
        .hysteresis_type = hyster_type, \
    }; \
    input_type limits[2] = {0}; \
    get_limits_ ## id(settings, threshold, limits); \
    if (settings.int_state) \
    { \
        if (data >= limits[1]) \
        { \
            return true; \
        } \
        settings.int_state = false; \
        return false; \
    } \
    if (data >= limits[0]) \
    { \
        settings.int_state = true; \
        return true; \
    } \
    return false; \
}
#endif//DEF_NEW_EVENTDETECTION_IMPL

#ifndef DEF_NEW_EVENTDETECTION_PROTO
#define DEF_NEW_EVENTDETECTION_PROTO(id, input_type) \
bool is_event_ ## id(input_type data, input_type threshold);
#endif//DEF_NEW_EVENTDETECTION_PROTO
#endif//EVENTDETECTION_TEMPLATE_H
