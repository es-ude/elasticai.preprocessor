#ifndef REPLAYER_TEMPLATE_H
#define REPLAYER_TEMPLATE_H

#include <stdint.h>

/* -------------------------------------------------------------------------
 * State structs
 *
 * ReplayerState      — data-only replayer (mirrors REPLAYER without ADD_TRIGGER)
 * ReplayerTrggState  — replayer with parallel trigger channel (ADD_TRIGGER)
 * ---------------------------------------------------------------------- */

typedef struct {
    uint16_t pos;         /* current read position (= cnt_pos in Verilog)  */
    uint16_t num_values;  /* total number of stored values (= NUM_VALUES)   */
    void    *data;        /* pointer to the data array (= bram_data)        */
} ReplayerState;

typedef struct {
    uint16_t  pos;
    uint16_t  num_values;
    void     *data;
    uint8_t  *trgg;       /* pointer to the trigger array (= bram_trgg)     */
} ReplayerTrggState;


/* =========================================================================
 * Variant A: data only  (no ADD_TRIGGER)
 * =========================================================================
 *
 * Generated API per instance ID:
 *
 *   data_type replayer_next_ID(void)
 *       Read the value at the current position and advance the counter.
 *       Wraps back to 0 after the last value — mirrors Verilog's always-block:
 *         cnt_pos <= (cnt_pos == NUM_VALUES-1) ? 0 : cnt_pos + 1
 *
 *   int replayer_done_ID(void)
 *       Returns 1 if the current position is the last value (pos == n-1).
 *       Mirrors: DATA_END = (cnt_pos == NUM_VALUES-1)
 *       IMPORTANT: call BEFORE replayer_next, not after — next advances pos.
 *
 *   void replayer_reset_ID(void)
 *       Reset counter to 0.  Mirrors: RSTN (active-low reset).
 *
 * Typical usage:
 *   while (1) {
 *       int done     = replayer_done_ID();   // check end BEFORE reading
 *       data_type v  = replayer_next_ID();   // read + advance
 *       process(v);
 *       if (done) break;
 *   }
 * ======================================================================= */

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


/* =========================================================================
 * Variant B: data + trigger  (ADD_TRIGGER)
 *
 * The trigger array must be declared BEFORE invoking DEF_NEW_REPLAYER_TRGG_IMPL.
 * The Python generator emits it automatically; when using the macro by hand:
 *
 *   static uint8_t my_trgg[] = {0, 0, 1, 0, ...};
 *   DEF_NEW_REPLAYER_TRGG_IMPL(0, int16_t, 19, my_trgg, 0x001, 0x002, ...)
 *
 * Additional API function:
 *
 *   uint8_t replayer_trgg_ID(void)
 *       Returns the trigger bit at the CURRENT position (before advancing).
 *       Mirrors: DATA_TRGG = bram_trgg[cnt_pos]
 *       Call this before replayer_next_ID() to get the trigger at the same
 *       position as the data value being read.
 * ======================================================================= */

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
