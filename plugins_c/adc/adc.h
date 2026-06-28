#ifndef ADC_H
#define ADC_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int   total_bits;  /* Total bitwidth (e.g. 12 for a 12-bit ADC)         */
    int   frac_bits;   /* Fractional bits (0 = pure integer output)          */
    int   is_signed;   /* 1 = signed output, 0 = unsigned output             */
    float srate_orig;  /* Input sampling rate [Hz]                           */
    float srate_new;   /* Target sampling rate [Hz]                          */
    float vneg;        /* Minimum measurable voltage [V]                     */
    float vpos;        /* Maximum measurable voltage [V]                     */
} SettingsADC;

extern const SettingsADC DefaultSettingsADC;

/* -------------------------------------------------------------------------
 * Computed properties (mirror of Python @property)
 * ---------------------------------------------------------------------- */

/* vcm: midpoint of the voltage range = (vpos + vneg) / 2 */
float adc_vcm(const SettingsADC *s);

/* lsb: voltage represented by one bit = (vpos - vneg) / 2^total_bits */
float adc_lsb(const SettingsADC *s);

/* min_step: smallest representable fixed-point step = 2^(-frac_bits) */
float adc_min_step(const SettingsADC *s);

/* -------------------------------------------------------------------------
 * Clamping
 * ---------------------------------------------------------------------- */

/* Clamp a voltage to [vneg, vpos]. */
float adc_clamp_analog(const SettingsADC *s, float v);

/* Clamp an integer sample to the representable integer range. */
int64_t adc_clamp_int(const SettingsADC *s, int64_t v);

/* Clamp a fixed-point rational to the representable rational range. */
float adc_clamp_fxp(const SettingsADC *s, float v);

/* -------------------------------------------------------------------------
 * Single-sample quantization
 *
 * adc_voltage_to_*: the core ADC conversion — voltage [V] → digital value.
 * adc_fxp_to_*:     re-quantize a fixed-point rational to a new representation.
 * adc_int_to_*:     re-quantize an integer sample to a new representation.
 * ---------------------------------------------------------------------- */

int64_t adc_voltage_to_int(const SettingsADC *s, float v);
float   adc_voltage_to_fxp(const SettingsADC *s, float v);

int64_t adc_fxp_to_int(const SettingsADC *s, float v);
float   adc_fxp_to_fxp(const SettingsADC *s, float v);

int64_t adc_int_to_int(const SettingsADC *s, int64_t v);
float   adc_int_to_fxp(const SettingsADC *s, int64_t v);

/* -------------------------------------------------------------------------
 * Resampling
 *
 * Uses linear interpolation with DC-offset removal (see _do_resample in
 * adc.py). scipy's polyphase FIR approach is more accurate but not
 * practical on MCU — the difference is negligible for slow biological signals.
 *
 * adc_resample_out_len: expected number of output samples (allocate before call).
 * adc_resample:         writes into caller-provided out[0..out_max-1].
 *                       Returns number of samples actually written.
 * ---------------------------------------------------------------------- */
size_t adc_resample_out_len(const SettingsADC *s, size_t in_len);

size_t adc_resample(const SettingsADC *s,
                    const float *in, size_t in_len,
                    float *out,      size_t out_max);

/* -------------------------------------------------------------------------
 * Time-range cutting
 *
 * adc_cut_transient: sets *out_start to point into in[] and returns the
 *   number of samples in [t_start, t_stop).  No data is copied.
 *   use_srate_orig=1 → convert time using srate_orig; 0 → srate_new.
 *   Pass t_start >= t_stop (or both 0) to return the full array unchanged.
 *
 * adc_cut_labels: finds the first label position at or after t_start and
 *   the first at or after t_stop.  Sets *out_offset to the start index and
 *   returns the count.  The caller reads label_id[*out_offset .. +count-1]
 *   and label_pos[*out_offset .. +count-1].
 * ---------------------------------------------------------------------- */
size_t adc_cut_transient(const SettingsADC *s,
                          const float *in, size_t in_len,
                          float t_start, float t_stop, int use_srate_orig,
                          const float **out_start);

size_t adc_cut_labels(const SettingsADC *s,
                       const size_t *label_pos, size_t n_labels,
                       float t_start, float t_stop, int use_srate_orig,
                       size_t *out_offset);

/* -------------------------------------------------------------------------
 * Full pipeline: resample + quantize
 *
 * The caller must provide a scratch buffer tmp[0..tmp_max-1] for the
 * resampled float data (size >= adc_resample_out_len(s, in_len)).
 * The integer results are written to out[].
 * Returns the number of samples written to out[].
 * ---------------------------------------------------------------------- */
size_t adc_redefine_from_voltage(const SettingsADC *s,
                                  const float   *in,  size_t in_len,
                                  float         *tmp, size_t tmp_max,
                                  int64_t       *out);

size_t adc_redefine_from_fxp(const SettingsADC *s,
                               const float   *in,  size_t in_len,
                               float         *tmp, size_t tmp_max,
                               int64_t       *out);

size_t adc_redefine_from_int(const SettingsADC *s,
                               const int64_t *in,  size_t in_len,
                               float         *tmp, size_t tmp_max,
                               int64_t       *out);

#endif /* ADC_H */
