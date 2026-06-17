#ifndef WAVEFORM_LUT_TEMPLATE_H
#define WAVEFORM_LUT_TEMPLATE_H
#include <stdint.h>
#include <stdbool.h>


typedef struct {
    uint8_t state;
    uint16_t lut_cnt;
    uint16_t lut_position;
    uint16_t lut_length;
    uint16_t lut_offset;
    void *lut_data;
} WaveformSettings;


#ifndef DEF_GET_LUT_VALUE_FULL
#define DEF_GET_LUT_VALUE_FULL(input_type, cnt_type) \
void rst_waveform_cnt_full (WaveformSettings *filter){ \
    filter->lut_position = 0; \
}; \
cnt_type get_waveform_pos_full(WaveformSettings *filter) { \
    return filter->lut_position; \
}; \
input_type read_waveform_value_runtime_full (WaveformSettings *filter, bool skip_last_point) { \
    input_type* lut_values = filter->lut_data; \
    \
    input_type data = filter->lut_offset + lut_values[filter->lut_position]; \
    if((filter->lut_position == filter->lut_length -1) && !skip_last_point){ \
        filter->lut_position = 0; \
        filter->state = 0; \
    } else if((filter->lut_position == filter->lut_length -2) && skip_last_point){ \
        filter->lut_position = 0; \
        filter->state = 0; \
    } else { \
        filter->lut_position++; \
        filter->state = 1; \
    }; \
    return data; \
};
#endif


#ifndef DEF_NEW_WAVEFORM_LUT_FULL_IMPL
#define DEF_NEW_WAVEFORM_LUT_FULL_IMPL(id, input_type, cnt_type, lut_lgth, offset, ...) \
    static DEF_GET_LUT_VALUE_FULL(input_type, cnt_type) \
    static input_type lut_data_read [] = {__VA_ARGS__}; \
    static WaveformSettings settings_## id = { \
        .state = 0, \
        .lut_cnt = 0, \
        .lut_position = 0, \
        .lut_length = lut_lgth, \
        .lut_offset = offset, \
        .lut_data = lut_data_read \
    }; \
    input_type get_waveform_value_ ## id (bool skip_last_point) { \
        return read_waveform_value_runtime_full(& (settings_## id), skip_last_point); \
    }; \
    cnt_type get_waveform_pos_## id (void) { \
        return settings_## id.lut_cnt; \
    }; \
    cnt_type get_waveform_lgth_## id (bool skip_last_point) { \
        return lut_lgth - ((skip_last_point) ?  2 : 1); \
    }; \
    void rst_waveform_cnt_## id (void){ \
        settings_## id.lut_cnt = 0; \
        settings_## id.lut_position = 0; \
        settings_## id.state = 0; \
    };
#endif


#ifndef DEF_GET_LUT_VALUE_OPT
#define DEF_GET_LUT_VALUE_OPT(input_type, cnt_type) \
input_type read_waveform_value_runtime_opt (WaveformSettings *filter, bool skip_last_point) { \
    input_type* lut_values = filter->lut_data; \
    input_type data = 0; \
    \
    if(filter->state == 0){ \
        data = filter->lut_offset + lut_values[filter->lut_position]; \
        filter->lut_position++; \
        filter->lut_cnt++; \
        if(filter->lut_position == (filter->lut_length - 1)){ \
            filter->state = 1; \
        } else { \
            filter->state = 0; \
        }; \
    } else if(filter->state == 1){ \
        data = filter->lut_offset + lut_values[filter->lut_position]; \
        filter->lut_position--; \
        filter->lut_cnt++; \
        if(filter->lut_position == 0){ \
            filter->state = 2; \
        } else { \
            filter->state = 1; \
        }; \
    } else if(filter->state == 2){ \
        data = filter->lut_offset - lut_values[filter->lut_position]; \
        filter->lut_position++; \
        filter->lut_cnt++; \
        if(filter->lut_position == (filter->lut_length - 1)){ \
            filter->state = 3; \
        } else { \
            filter->state = 2; \
        }; \
    } else if(filter->state == 3){ \
        data = filter->lut_offset - lut_values[filter->lut_position]; \
        filter->lut_position--; \
        filter->lut_cnt++; \
        if(filter->lut_position == 0){ \
            if(skip_last_point){ \
                filter->state = 0; \
            } else { \
                filter->state = 4; \
            } \
        } else { \
            filter->state = 3; \
        }; \
    } else if(filter->state == 4){ \
        data = filter->lut_offset; \
        filter->lut_position = 0; \
        filter->lut_cnt = 0; \
        filter->state = 0; \
    } else { \
        data = filter->lut_offset; \
        filter->lut_position = 0; \
        filter->lut_cnt = 0; \
        filter->state = 0; \
    }; \
    return data; \
};
#endif


#ifndef DEF_NEW_WAVEFORM_LUT_OPT_IMPL
#define DEF_NEW_WAVEFORM_LUT_OPT_IMPL(id, input_type, cnt_type, lut_lgth, offset, ...) \
    static DEF_GET_LUT_VALUE_OPT(input_type, cnt_type) \
    static input_type lut_data_read [] = {__VA_ARGS__}; \
    static WaveformSettings settings_## id = { \
        .state = 0, \
        .lut_cnt = 0, \
        .lut_position = 0, \
        .lut_length = lut_lgth, \
        .lut_offset = offset, \
        .lut_data = lut_data_read \
    }; \
    input_type get_waveform_value_ ## id (bool skip_last_point) { \
        return read_waveform_value_runtime_opt(& (settings_## id), skip_last_point); \
    }; \
    cnt_type get_waveform_pos_## id (void) { \
        return settings_## id.lut_cnt; \
    }; \
    cnt_type get_waveform_lgth_## id (bool skip_last_point) { \
        return 4* lut_lgth - ((skip_last_point) ? 5 : 4); \
    }; \
    void rst_waveform_cnt_## id(void){ \
        settings_## id.lut_cnt = 0; \
        settings_## id.lut_position = 0; \
        settings_## id.state = 0; \
    };
#endif


#ifndef DEF_NEW_WAVEFORM_LUT_PROTO
#define DEF_NEW_WAVEFORM_LUT_PROTO(id, input_type, cnt_type) \
    input_type get_waveform_value_ ## id (bool skip_last_point); \
    cnt_type get_waveform_lgth_## id (bool skip_last_point); \
    cnt_type get_waveform_pos_## id (void); \
    void rst_waveform_cnt_## id(void);
#endif


#endif
