#include "downsampling.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* -------------------------------------------------------------------------
 * default-setting
 * ---------------------------------------------------------------------- */
const SettingsDownSampling DefaultSettingsDownSampling = {
    .sampling_rate = 1000.0f,
    .dsr           = 10,
};

/* -------------------------------------------------------------------------
 * helpers
 * ---------------------------------------------------------------------- */
static int is_power_of_two(int n)
{
    return n >= 2 && (n & (n - 1)) == 0;
}

float sampling_rate_out(const SettingsDownSampling *s)
{
    return s->sampling_rate / (float)s->dsr;
}

/* -------------------------------------------------------------------------
 * do_simple: average every dsr input samples into one output sample.
 * ---------------------------------------------------------------------- */

void do_simple(
    const SettingsDownSampling *s,
    const float *uin, 
    size_t uin_len,
    float *uout)
{
    size_t dsr      = s->dsr;
    size_t sz       = uin_len / (size_t)dsr; 

    for (size_t i = 0; i < sz; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < dsr; j++) {
            sum += uin[i * dsr + j];
        }
        uout[i] = (sum / (float)dsr);
    }
}

/* -------------------------------------------------------------------------
 * do_cic: CIC-filter downsampling (Cascaded Integrator-Comb)
 * with helpers
 *
 * Internally uses int64_t fixed-point to avoid float accumulation drift.
 * The scale factor Q (a power of 2) is chosen so that the integrator state
 * never exceeds INT64_MAX: Q = 2^(62 - ceil(N * log2(dsr))).
 * ---------------------------------------------------------------------- */
typedef struct { int64_t yn; int64_t ynm; } Integrator;
typedef struct { int64_t xn; int64_t xnm; } Comb;

static int64_t integrator_update(Integrator *inte, int64_t inp)
{
    inte->ynm = inte->yn;
    inte->yn  = inte->ynm + inp;
    return inte->yn;
}

static int64_t comb_update(Comb *c, int64_t inp)
{
    c->xnm = c->xn;
    c->xn  = inp;
    return c->xn - c->xnm;
}


void do_cic(
    const SettingsDownSampling *s,
    const float *uin,
    size_t uin_len,
    int num_stages,
    float *uout)
{
    size_t dsr = (size_t)s->dsr;

    /* bits consumed by CIC growth: integrators accumulate up to input * dsr^N */
    int growth_bits = (int)ceilf((float)num_stages * log2f((float)dsr));
    int frac_bits   = 62 - growth_bits;
    if (frac_bits < 1) frac_bits = 1;
    int64_t Q = (int64_t)1 << frac_bits;

    /* exact integer gain = dsr^num_stages */
    int64_t gain = 1;
    for (int i = 0; i < num_stages; i++) gain *= (int64_t)dsr;

    Integrator *intes = calloc((size_t)num_stages, sizeof(Integrator));
    Comb       *combs = calloc((size_t)num_stages, sizeof(Comb));
    if (!intes || !combs) { free(intes); free(combs); return; }

    size_t out_idx = 0;
    for (size_t idx = 0; idx < uin_len; idx++) {
        int64_t z = (int64_t)(uin[idx] * (float)Q);
        for (int i = 0; i < num_stages; i++)
            z = integrator_update(&intes[i], z);

        if (idx % dsr == 0) {
            for (int c = 0; c < num_stages; c++)
                z = comb_update(&combs[c], z);
            uout[out_idx++] = (float)z / ((float)Q * (float)gain);
        }
    }
    free(intes);
    free(combs);
}

/* -------------------------------------------------------------------------
 * do_decimation_polyphase_order_one: First-order polyphase decimation by factor 2.
 * FIR coefficients: [1, 1] 
 * ---------------------------------------------------------------------- */
void do_decimation_polyphase_order_one(
    const float *uin, 
    size_t uin_len,
    float *uout)
{
    float  last_hs  = 0.0f;
    size_t out_idx  = 0;

    for (size_t idx = 0; idx < uin_len; idx++) {
        float val = uin[idx];
        if (idx % 2 == 1) {
            uout[out_idx++] = val + last_hs;
        }
        last_hs = val;
    }
}

/* -------------------------------------------------------------------------
 * do_decimation_polyphase_order_two
 * ---------------------------------------------------------------------- */
void do_decimation_polyphase_order_two(
    const float *uin, 
    size_t uin_len,
    float *uout)
{
    float  last_hs  = 0.0f;
    float  last_ls  = 0.0f;
    size_t out_idx  = 0;

    for (size_t idx = 0; idx < uin_len; idx++) {
        float val = uin[idx];
        if (idx % 2 == 1) {
            uout[out_idx++] = val + last_ls + 2.0f * last_hs;
            last_ls = val;
        }
        last_hs = val;
    }
}

/* -------------------------------------------------------------------------
 * do_decimation_polyphase
 * ---------------------------------------------------------------------- */
void do_decimation_polyphase(
    const SettingsDownSampling *s,
    const float *uin, 
    size_t uin_len,
    int take_first_order,
    float *uout)
{
    if (!is_power_of_two(s->dsr)) return;

    int steps = 0;
    int n = s->dsr;
    while (n > 1) { n >>= 1; steps++; }
    if (steps == 0) return;

    /* Einzelstufe: direkt in den vom Aufrufer bereitgestellten Puffer schreiben */
    if (steps == 1) {
        if (take_first_order)
            do_decimation_polyphase_order_one(uin, uin_len, uout);
        else
            do_decimation_polyphase_order_two(uin, uin_len, uout);
        return;
    }

    /* Mehrstufig: zwei Ping-Pong-Puffer für interne Zwischenergebnisse.
     * Größe uin_len/2 reicht für alle Zwischenschritte, da jede Stufe
     * die Datenmenge halbiert. */
    float *bufs[2] = {
        malloc((uin_len / 2) * sizeof(float)),
        malloc((uin_len / 2) * sizeof(float))
    };
    if (!bufs[0] || !bufs[1]) { free(bufs[0]); free(bufs[1]); return; }

    const float *src      = uin;
    size_t       src_size = uin_len;
    int          cur      = 0;

    for (int i = 0; i < steps; i++) {
        /* Letzter Schritt schreibt direkt in den Aufrufer-Puffer */
        float *dst = (i == steps - 1) ? uout : bufs[cur];

        if (take_first_order)
            do_decimation_polyphase_order_one(src, src_size, dst);
        else
            do_decimation_polyphase_order_two(src, src_size, dst);

        src      = dst;
        src_size = src_size / 2;
        cur     ^= 1;
    }

    free(bufs[0]);
    free(bufs[1]);
}
