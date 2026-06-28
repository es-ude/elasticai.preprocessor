#include "adc.h"

#include <math.h>
#include <stddef.h>

/* -------------------------------------------------------------------------
 * Default settings
 * ---------------------------------------------------------------------- */
const SettingsADC DefaultSettingsADC = {
    .total_bits = 12,
    .frac_bits  = 0,
    .is_signed  = 1,
    .srate_orig = 1000.0f,
    .srate_new  = 1000.0f,
    .vneg       = -1.0f,
    .vpos       =  1.0f,
};

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/* Smallest and largest representable integer values for this settings. */
static int64_t min_int(const SettingsADC *s)
{
    return s->is_signed ? -((int64_t)1 << (s->total_bits - 1)) : 0;
}

static int64_t max_int(const SettingsADC *s)
{
    return s->is_signed
        ? (((int64_t)1 << (s->total_bits - 1)) - 1)
        : (((int64_t)1 << s->total_bits) - 1);
}

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static int64_t clampi(int64_t v, int64_t lo, int64_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* -------------------------------------------------------------------------
 * Computed properties
 * ---------------------------------------------------------------------- */

float adc_vcm(const SettingsADC *s)
{
    return (s->vpos + s->vneg) * 0.5f;
}

float adc_lsb(const SettingsADC *s)
{
    return (s->vpos - s->vneg) / (float)((int64_t)1 << s->total_bits);
}

float adc_min_step(const SettingsADC *s)
{
    return 1.0f / (float)((int64_t)1 << s->frac_bits);
}

/* -------------------------------------------------------------------------
 * Clamping
 * ---------------------------------------------------------------------- */

float adc_clamp_analog(const SettingsADC *s, float v)
{
    return clampf(v, s->vneg, s->vpos);
}

int64_t adc_clamp_int(const SettingsADC *s, int64_t v)
{
    return clampi(v, min_int(s), max_int(s));
}

float adc_clamp_fxp(const SettingsADC *s, float v)
{
    float step = adc_min_step(s);
    return clampf(v, (float)min_int(s) * step, (float)max_int(s) * step);
}

/* -------------------------------------------------------------------------
 * Single-sample quantization
 *
 * Core formula (mirrors _quantize_voltage in adc.py):
 *   step_count = round((clamp(v, vneg, vpos) - vneg) / lsb)
 *   int_val    = clamp(step_count + min_int, min_int, max_int)
 *   fxp_val    = int_val * min_step
 * ---------------------------------------------------------------------- */

int64_t adc_voltage_to_int(const SettingsADC *s, float v)
{
    float lsb      = adc_lsb(s);
    float clamped  = adc_clamp_analog(s, v);
    int64_t steps  = (int64_t)roundf((clamped - s->vneg) / lsb);
    return clampi(steps + min_int(s), min_int(s), max_int(s));
}

float adc_voltage_to_fxp(const SettingsADC *s, float v)
{
    return (float)adc_voltage_to_int(s, v) * adc_min_step(s);
}

/* Re-quantize a fixed-point rational value.
 * Mirrors _quantize_digital(..., is_int_input=False). */
int64_t adc_fxp_to_int(const SettingsADC *s, float v)
{
    float step    = adc_min_step(s);
    float clamped = adc_clamp_fxp(s, v);
    return (int64_t)roundf(clamped / step);
}

float adc_fxp_to_fxp(const SettingsADC *s, float v)
{
    return (float)adc_fxp_to_int(s, v) * adc_min_step(s);
}

/* Re-quantize an integer value.
 * Mirrors _quantize_digital(..., is_int_input=True). */
int64_t adc_int_to_int(const SettingsADC *s, int64_t v)
{
    return clampi(v, min_int(s), max_int(s));
}

float adc_int_to_fxp(const SettingsADC *s, int64_t v)
{
    return (float)adc_int_to_int(s, v) * adc_min_step(s);
}

/* -------------------------------------------------------------------------
 * Resampling
 *
 * Python uses scipy.signal.resample_poly, which applies a polyphase FIR
 * anti-aliasing filter.  Here we use linear interpolation — simpler and
 * sufficient for slow biological signals on MCU.
 *
 * Like the Python version, the DC offset (first sample) is removed before
 * interpolation and added back afterwards, preventing filter edge effects.
 * ---------------------------------------------------------------------- */

size_t adc_resample_out_len(const SettingsADC *s, size_t in_len)
{
    if (s->srate_new == 0.0f || s->srate_orig == 0.0f)
        return in_len;
    return (size_t)((float)in_len * s->srate_new / s->srate_orig);
}

size_t adc_resample(const SettingsADC *s,
                    const float *in, size_t in_len,
                    float *out,      size_t out_max)
{
    /* No resampling needed */
    if (s->srate_new == 0.0f || in_len <= 1 ||
        s->srate_new == s->srate_orig) {
        size_t n = in_len < out_max ? in_len : out_max;
        for (size_t i = 0; i < n; i++)
            out[i] = in[i];
        return n;
    }

    float ratio    = s->srate_orig / s->srate_new; /* input samples per output sample */
    size_t out_len = adc_resample_out_len(s, in_len);
    if (out_len > out_max) out_len = out_max;

    float xoff = in[0]; /* DC offset removal — matches Python _do_resample */

    for (size_t i = 0; i < out_len; i++) {
        float pos = (float)i * ratio;
        size_t j  = (size_t)pos;
        float frc = pos - (float)j;

        float a = in[j] - xoff;
        float b = (j + 1 < in_len) ? (in[j + 1] - xoff) : a;

        out[i] = a + frc * (b - a) + xoff;
    }
    return out_len;
}

/* -------------------------------------------------------------------------
 * Time-range cutting
 * ---------------------------------------------------------------------- */

size_t adc_cut_transient(const SettingsADC *s,
                          const float *in, size_t in_len,
                          float t_start, float t_stop, int use_srate_orig,
                          const float **out_start)
{
    /* No cut: return full array */
    if (t_start >= t_stop) {
        *out_start = in;
        return in_len;
    }

    float srate = use_srate_orig ? s->srate_orig : s->srate_new;

    size_t idx0 = (size_t)(t_start * srate);
    size_t idx1 = (size_t)(t_stop  * srate);

    if (idx0 >= in_len) idx0 = in_len;
    if (idx1 >= in_len) idx1 = in_len;

    *out_start = in + idx0;
    return idx1 - idx0;
}

size_t adc_cut_labels(const SettingsADC *s,
                       const size_t *label_pos, size_t n_labels,
                       float t_start, float t_stop, int use_srate_orig,
                       size_t *out_offset)
{
    /* No cut: return all labels */
    if (t_start >= t_stop) {
        *out_offset = 0;
        return n_labels;
    }

    float srate = use_srate_orig ? s->srate_orig : s->srate_new;

    /* Keep thresholds as float to match Python's time-domain comparison:
     *   first label whose label_pos/srate >= t_start/t_stop.
     * Integer truncation of t*srate would give a different boundary when
     * t*srate is not an integer (e.g. 0.3*5 = 1.5 truncates to 1, but the
     * correct threshold is 1.5, skipping label_pos=1 which is at time 0.2). */
    float fpos0 = t_start * srate;
    float fpos1 = t_stop  * srate;

    /* Find first label at or after t_start */
    size_t i0 = 0;
    while (i0 < n_labels && (float)label_pos[i0] < fpos0)
        i0++;

    /* Find first label at or after t_stop */
    size_t i1 = i0;
    while (i1 < n_labels && (float)label_pos[i1] < fpos1)
        i1++;

    *out_offset = i0;
    return i1 - i0;
}

/* -------------------------------------------------------------------------
 * Full pipeline: resample + quantize
 *
 * All three variants resample into the caller-provided tmp[] buffer, then
 * quantize element by element into out[].  The function returns the number
 * of output samples.
 * ---------------------------------------------------------------------- */

size_t adc_redefine_from_voltage(const SettingsADC *s,
                                  const float   *in,  size_t in_len,
                                  float         *tmp, size_t tmp_max,
                                  int64_t       *out)
{
    size_t n = adc_resample(s, in, in_len, tmp, tmp_max);
    for (size_t i = 0; i < n; i++)
        out[i] = adc_voltage_to_int(s, tmp[i]);
    return n;
}

size_t adc_redefine_from_fxp(const SettingsADC *s,
                               const float   *in,  size_t in_len,
                               float         *tmp, size_t tmp_max,
                               int64_t       *out)
{
    size_t n = adc_resample(s, in, in_len, tmp, tmp_max);
    for (size_t i = 0; i < n; i++)
        out[i] = adc_fxp_to_int(s, tmp[i]);
    return n;
}

size_t adc_redefine_from_int(const SettingsADC *s,
                               const int64_t *in,  size_t in_len,
                               float         *tmp, size_t tmp_max,
                               int64_t       *out)
{
    /* Convert int → float, resample as float, then re-quantize. */
    float step = adc_min_step(s);
    size_t n_in = in_len < tmp_max ? in_len : tmp_max;
    for (size_t i = 0; i < n_in; i++)
        tmp[i] = (float)in[i] * step;

    float *resampled = tmp + n_in; /* use second half of tmp as resampled buffer */
    size_t resampled_max = tmp_max - n_in;
    size_t n = adc_resample(s, tmp, n_in, resampled, resampled_max);

    for (size_t i = 0; i < n; i++)
        out[i] = adc_fxp_to_int(s, resampled[i]);
    return n;
}
