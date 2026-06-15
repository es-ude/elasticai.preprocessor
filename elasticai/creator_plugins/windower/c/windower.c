#include "windower.h"

void windower_init(Windower* w, float* buffer, int window_size) {
    w->buffer = buffer;
    w->window_size = window_size;
    w->count = 0;
}

void windower_push(Windower* w, float sample) {
    int i;

    if (w->count < w->window_size) {
        w->buffer[w->count] = sample;
        w->count += 1;
        return;
    }

    /* shift left, drop oldest */
    for (i = 1; i < w->window_size; i++) {
        w->buffer[i - 1] = w->buffer[i];
    }

    w->buffer[w->window_size - 1] = sample;
}

int windower_is_full(Windower* w) {
    return w->count >= w->window_size;
}

int windower_get_window(Windower* w, float* out) {
    int i;

    if (!windower_is_full(w)) {
        return 0;
    }

    for (i = 0; i < w->window_size; i++) {
        out[i] = w->buffer[i];
    }

    return 1;
}

void windower_reset(Windower* w) {
    w->count = 0;
}
