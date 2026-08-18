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
    input_type threshold; \
    float      hysteresis; \
    bool       int_state; \
    EventDetectionHysteresisType hysteresis_type; \
} EventSettings_ ## id;
#endif//EVENT_SETTINGS

#ifndef TYPE_HYSTERESIS
#define TYPE_HYSTERESIS(id, input_type) \
void get_limits_ ## id(EventSettings_ ## id settings, input_type *limits) { \
    input_type thr_zero = settings.threshold; \
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

#ifndef DEF_EVENTDETECTION
#define DEF_EVENTDETECTION(id, input_type) \
bool is_still_event_ ## id(input_type data, bool is_event, input_type *limits) { \
    if (is_event) \
        { \
            return (data >= limits[1]); \
        } \
    return (data >= limits[0]); \
}
#endif//DEF_EVENTDETECTION

#ifndef DEF_NEW_EVENTDETECTION_IMPL
#define DEF_NEW_EVENTDETECTION_IMPL(id, input_type, thresh, hyster, hyster_type) \
static DEF_EVENTDETECTION(id, input_type) \
bool is_event_ ## id(input_type data) {  \
    static EventSettings_ ## id settings = { \
        .threshold       = thresh, \
        .hysteresis      = hyster, \
        .int_state       = false, \
        .hysteresis_type = hyster_type, \
    }; \
    static input_type limits[2] = {0}; \
    get_limits_ ## id(settings, limits); \
    if (is_still_event_ ## id(data, settings.int_state, limits)) { \
        settings.int_state = true; \
        return true; \
    } \
    settings.int_state = false; \
    return false; \
}
#endif//DEF_NEW_EVENTDETECTION_IMPL

#ifndef DEF_NEW_EVENTDETECTION_PROTO
#define DEF_NEW_EVENTDETECTION_PROTO(id, input_type) \
bool is_event_ ## id(input_type data);
#endif//DEF_NEW_EVENTDETECTION_PROTO
#endif//EVENTDETECTION_TEMPLATE_H
