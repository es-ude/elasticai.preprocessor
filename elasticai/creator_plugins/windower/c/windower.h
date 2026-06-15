typedef struct {
    float* buffer;
    int window_size;
    int count;
} Windower;

void windower_init(Windower* w, float* buffer, int window_size);
void windower_push(Windower* w, float sample);
int windower_is_full(Windower* w);
int windower_get_window(Windower* w, float* out);
void windower_reset(Windower* w);
