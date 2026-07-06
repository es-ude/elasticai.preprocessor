#ifndef REPLAYER_TEMPLATE_H
#define REPLAYER_TEMPLATE_H

#include <stdint.h>

// State structs:
// ReplayerState      — data-only replayer
// ReplayerTrggState  — replayer with parallel trigger channel

typedef struct {
    uint16_t pos;         // current read position
    uint16_t num_values;  // total number of stored values
    void    *data;        // pointer to the data array
} ReplayerState;

typedef struct {
    uint16_t  pos;
    uint16_t  num_values;
    void     *data;
    uint8_t  *trgg;       // pointer to the trigger array
} ReplayerTrggState;


// data only
// generates tree functions:
//
//   data_type replayer_next_ID(void)
//   read current positions value and increase counter
//
//   int replayer_done_ID(void)
//   returns 1 if current position is the last value
//
//   void replayer_reset_ID(void)
//   reset counter to 0.
//
// typical use: 
//      wile (1) 
//      {
//          int done    = replayer_done_ID();
//          data_type v = replayer_next_ID();
//          process(v);
//          if (done) break;
//      }

#ifndef DEF_REPLAYER_READ
#define DEF_REPLAYER_READ(id, data_type) \
data_type replayer_read_##id(ReplayerState *r) { \
    data_type *arr = (data_type *)(r->data); \
    data_type  val = arr[r->pos]; \
    r->pos = (uint16_t)((r->pos == (uint16_t)(r->num_values - 1u)) \
                        ? 0u : r->pos + 1u); \
    return val; \
}
#endif


#ifndef DEF_NEW_REPLAYER_IMPL
/* n_vals is intentionally different from the struct field 'num_values' to
 * avoid the C preprocessor substituting the token inside .num_values = ... */
#define DEF_NEW_REPLAYER_IMPL(id, data_type, n_vals, ...) \
    static DEF_REPLAYER_READ(id, data_type) \
    static data_type      replayer_data_##id[]  = {__VA_ARGS__}; \
    static ReplayerState  replayer_state_##id   = { \
        .pos        = 0u, \
        .num_values = (n_vals), \
        .data       = replayer_data_##id, \
    }; \
    data_type replayer_next_##id(void) { \
        return replayer_read_##id(&replayer_state_##id); \
    } \
    int replayer_done_##id(void) { \
        return (int)(replayer_state_##id.pos \
                     == (uint16_t)(replayer_state_##id.num_values - 1u)); \
    } \
    void replayer_reset_##id(void) { \
        replayer_state_##id.pos = 0u; \
    }
#endif


#ifndef DEF_NEW_REPLAYER_PROTO
#define DEF_NEW_REPLAYER_PROTO(id, data_type) \
    data_type replayer_next_##id(void); \
    int       replayer_done_##id(void); \
    void      replayer_reset_##id(void);
#endif


// data and trigger
// additional function:
// uint8_t replayer_trgg_ID()
// returns trigger-bit at current position
// call before replayer_next_ID() to get trigger of correct position. 

#ifndef DEF_REPLAYER_TRGG_READ
#define DEF_REPLAYER_TRGG_READ(id, data_type) \
data_type replayer_trgg_read_##id(ReplayerTrggState *r) { \
    data_type *arr = (data_type *)(r->data); \
    data_type  val = arr[r->pos]; \
    r->pos = (uint16_t)((r->pos == (uint16_t)(r->num_values - 1u)) \
                        ? 0u : r->pos + 1u); \
    return val; \
}
#endif


#ifndef DEF_NEW_REPLAYER_TRGG_IMPL
#define DEF_NEW_REPLAYER_TRGG_IMPL(id, data_type, n_vals, trgg_array, ...) \
    static DEF_REPLAYER_TRGG_READ(id, data_type) \
    static data_type          replayer_data_##id[]  = {__VA_ARGS__}; \
    static ReplayerTrggState  replayer_state_##id   = { \
        .pos        = 0u, \
        .num_values = (n_vals), \
        .data       = replayer_data_##id, \
        .trgg       = (trgg_array), \
    }; \
    data_type replayer_next_##id(void) { \
        return replayer_trgg_read_##id(&replayer_state_##id); \
    } \
    uint8_t replayer_trgg_##id(void) { \
        return replayer_state_##id.trgg[replayer_state_##id.pos]; \
    } \
    int replayer_done_##id(void) { \
        return (int)(replayer_state_##id.pos \
                     == (uint16_t)(replayer_state_##id.num_values - 1u)); \
    } \
    void replayer_reset_##id(void) { \
        replayer_state_##id.pos = 0u; \
    }
#endif


#ifndef DEF_NEW_REPLAYER_TRGG_PROTO
#define DEF_NEW_REPLAYER_TRGG_PROTO(id, data_type) \
    data_type replayer_next_##id(void); \
    uint8_t   replayer_trgg_##id(void); \
    int       replayer_done_##id(void); \
    void      replayer_reset_##id(void);
#endif


#endif /* REPLAYER_TEMPLATE_H */
