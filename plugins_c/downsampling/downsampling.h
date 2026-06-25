#ifndef DOWNSAMPLING_H
#define DOWNSAMPLING_H

#include <stddef.h>

typedef struct {
    float sampling_rate; /* Input sampling rate [Hz] */
    int   dsr;           /* Downsampling ratio       */
} SettingsDownSampling;

extern const SettingsDownSampling DefaultSettingsDownSampling;

/* Returns the output sampling rate: SR_in / dsr. */
float sampling_rate_out(const SettingsDownSampling *s);

/* -------------------------------------------------------------------------
 * do_simple: average every dsr input samples into one output sample.
 * ---------------------------------------------------------------------- */
void do_simple(
    const SettingsDownSampling *s,
    const float *uin, 
    size_t uin_len,
    float *uout);

/* -------------------------------------------------------------------------
 * do_cic: CIC-filter downsampling (Cascaded Integrator-Comb)
 * with helpers
 * ---------------------------------------------------------------------- */
void do_cic(
    const SettingsDownSampling *s,
    const float *uin, 
    size_t uin_len,
    int num_stages,
    float *uout);

/* -------------------------------------------------------------------------
 * do_decimation_polyphase_order_one: First-order polyphase decimation by factor 2.
 * FIR coefficients: [1, 1] 
 * ---------------------------------------------------------------------- */
void do_decimation_polyphase_order_one(
    const float *uin, 
    size_t uin_len,
    float *uout);

/* -------------------------------------------------------------------------
 * do_decimation_polyphase_order_two: Second-order polyphase decimation by factor 2.
 * FIR coefficients: [1, 2, 1] 
 * ---------------------------------------------------------------------- */
void do_decimation_polyphase_order_two(
    const float *uin, 
    size_t uin_len,
    float *uout);

/* -------------------------------------------------------------------------
 * do_decimation_polyphase: Iterative polyphase decimation for any power-of-2 dsr.
 * Applies order_one or order_two log2(dsr) times in sequence.
 * ---------------------------------------------------------------------- */
void do_decimation_polyphase(
    const SettingsDownSampling *s,
    const float *uin, 
    size_t uin_len,
    int take_first_order,
    float *uout);

#endif /* DOWNSAMPLING_H */
