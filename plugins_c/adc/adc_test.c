/* adc_test.c  –  mirrors Python elasticai/preprocessor/adc/adc_test.py
 *
 * Compile:
 *   gcc -std=c11 -Wall -Wextra -o adc_test adc_test.c adc.c -lm
 * Run:
 *   ./adc_test
 */
#include "adc.h"

#include <math.h>
#include <stdio.h>
#include <stddef.h>

/* ─── minimal test harness (same style as downsampling_test.c) ─────────── */

static int g_run  = 0;
static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do {                                          \
    g_run++;                                                           \
    printf("%d. check:\t", g_run);                                     \
    if (cond) {                                                        \
        g_pass++;                                                      \
        printf("%s passed.\n", (msg));                                 \
    } else {                                                           \
        g_fail++;                                                      \
        fprintf(stderr, "FAIL  %-52s  (%s:%d)\n",                     \
                (msg), __FILE__, __LINE__);                            \
    }                                                                  \
} while (0)

#define CHECK_FEQ(a, b, eps, msg) \
    CHECK(fabsf((float)(a) - (float)(b)) <= (float)(eps), (msg))

#define CHECK_IEQ(a, b, msg) \
    CHECK((int64_t)(a) == (int64_t)(b), (msg))

/* ─── shared settings (mirrors Python adc_sets fixture) ────────────────── */

static SettingsADC g_sets;

static void setup(void)
{
    g_sets.total_bits = 8;
    g_sets.frac_bits  = 4;
    g_sets.is_signed  = 0;   /* unsigned */
    g_sets.srate_orig = 100.0f;
    g_sets.srate_new  = 100.0f;
    g_sets.vneg       = 0.0f;
    g_sets.vpos       = 0.0f;
}

/* ─── test_adc_settings_vcm ────────────────────────────────────────────── */

static void test_vcm(void)
{
    /* (vss=0.0, vdd=1.0, expected=0.5) */
    SettingsADC s = g_sets;
    s.vneg = 0.0f; s.vpos = 1.0f;
    CHECK_FEQ(adc_vcm(&s), 0.5f, 1e-6f, "vcm: 0..1 => 0.5");

    /* (vss=-3.3, vdd=+3.3, expected=0.0) */
    s.vneg = -3.3f; s.vpos = 3.3f;
    CHECK_FEQ(adc_vcm(&s), 0.0f, 1e-6f, "vcm: -3.3..+3.3 => 0.0");
}

/* ─── test_adc_settings_lsb ────────────────────────────────────────────── */

static void test_lsb(void)
{
    /* (bitwidth=4, vss=0.0, vdd=1.0, expected=0.0625) */
    SettingsADC s = g_sets;
    s.total_bits = 4; s.vneg = 0.0f; s.vpos = 1.0f;
    CHECK_FEQ(adc_lsb(&s), 0.0625f, 1e-7f, "lsb: 4-bit 0..1 => 0.0625");

    /* (bitwidth=8, vss=-3.3, vdd=+3.3, expected=0.02578125) */
    s.total_bits = 8; s.vneg = -3.3f; s.vpos = 3.3f;
    CHECK_FEQ(adc_lsb(&s), 0.02578125f, 1e-7f, "lsb: 8-bit -3.3..+3.3 => 0.02578125");
}

/* ─── test_adc_clamp_voltage ────────────────────────────────────────────── */

static void test_clamp_analog(void)
{
    /* (vpos=2.0, vneg=-2.0, ...) */
    SettingsADC s = g_sets;
    s.vneg = -2.0f; s.vpos = 2.0f;
    CHECK_FEQ(adc_clamp_analog(&s, -2.5f), -2.0f, 1e-7f, "clamp_analog: -2.5 => -2.0");
    CHECK_FEQ(adc_clamp_analog(&s, -2.0f), -2.0f, 1e-7f, "clamp_analog: -2.0 => -2.0");
    CHECK_FEQ(adc_clamp_analog(&s, -0.25f),-0.25f,1e-7f, "clamp_analog: -0.25 unchanged");
    CHECK_FEQ(adc_clamp_analog(&s,  1.5f),  1.5f, 1e-7f, "clamp_analog:  1.5 unchanged");
    CHECK_FEQ(adc_clamp_analog(&s,  2.0f),  2.0f, 1e-7f, "clamp_analog:  2.0 => 2.0");

    /* (vpos=1.0, vneg=-1.0, ...) */
    s.vneg = -1.0f; s.vpos = 1.0f;
    CHECK_FEQ(adc_clamp_analog(&s, 2.0f),  1.0f, 1e-7f, "clamp_analog: 2.0 => vpos 1.0");
    CHECK_FEQ(adc_clamp_analog(&s, 3.75f), 1.0f, 1e-7f, "clamp_analog: 3.75 => vpos 1.0");
}

/* ─── test_adc_clamp_integer ────────────────────────────────────────────── */

static void test_clamp_int(void)
{
    /* (bitwidth=8, fracwidth=0, is_signed=True, ...) */
    SettingsADC s = g_sets;
    s.total_bits = 8; s.frac_bits = 0; s.is_signed = 1;
    CHECK_IEQ(adc_clamp_int(&s, -512),  -128, "clamp_int signed: -512 => -128");
    CHECK_IEQ(adc_clamp_int(&s, -126),  -126, "clamp_int signed: -126 unchanged");
    CHECK_IEQ(adc_clamp_int(&s,    0),     0, "clamp_int signed: 0 unchanged");
    CHECK_IEQ(adc_clamp_int(&s,   64),    64, "clamp_int signed: 64 unchanged");
    CHECK_IEQ(adc_clamp_int(&s,  128),   127, "clamp_int signed: 128 => 127");
    CHECK_IEQ(adc_clamp_int(&s,  130),   127, "clamp_int signed: 130 => 127");
    CHECK_IEQ(adc_clamp_int(&s,  300),   127, "clamp_int signed: 300 => 127");

    /* (bitwidth=8, fracwidth=0, is_signed=False, ...) */
    s.is_signed = 0;
    CHECK_IEQ(adc_clamp_int(&s, -512),     0, "clamp_int unsigned: -512 => 0");
    CHECK_IEQ(adc_clamp_int(&s, -126),     0, "clamp_int unsigned: -126 => 0");
    CHECK_IEQ(adc_clamp_int(&s,    0),     0, "clamp_int unsigned: 0 unchanged");
    CHECK_IEQ(adc_clamp_int(&s,   64),    64, "clamp_int unsigned: 64 unchanged");
    CHECK_IEQ(adc_clamp_int(&s,  128),   128, "clamp_int unsigned: 128 unchanged");
    CHECK_IEQ(adc_clamp_int(&s,  130),   130, "clamp_int unsigned: 130 unchanged");
    CHECK_IEQ(adc_clamp_int(&s,  300),   255, "clamp_int unsigned: 300 => 255");
}

/* ─── test_adc_clamp_fxp ────────────────────────────────────────────────── */

static void test_clamp_fxp(void)
{
    SettingsADC s = g_sets;

    /* (bitwidth=4, fracwidth=2, is_signed=True, ...) min=-2.0, max=1.75 */
    s.total_bits = 4; s.frac_bits = 2; s.is_signed = 1;
    CHECK_FEQ(adc_clamp_fxp(&s, -2.5f), -2.0f, 1e-6f, "clamp_fxp signed: -2.5 => -2.0");
    CHECK_FEQ(adc_clamp_fxp(&s, -2.0f), -2.0f, 1e-6f, "clamp_fxp signed: -2.0 => -2.0");
    CHECK_FEQ(adc_clamp_fxp(&s,-0.25f), -0.25f,1e-6f, "clamp_fxp signed: -0.25 unchanged");
    CHECK_FEQ(adc_clamp_fxp(&s,  1.5f),  1.5f, 1e-6f, "clamp_fxp signed:  1.5 unchanged");
    CHECK_FEQ(adc_clamp_fxp(&s,  2.0f),  1.75f,1e-6f, "clamp_fxp signed:  2.0 => 1.75");

    /* (bitwidth=4, fracwidth=2, is_signed=False, ...) min=0.0, max=3.75 */
    s.is_signed = 0;
    CHECK_FEQ(adc_clamp_fxp(&s, -1.0f),  0.0f, 1e-6f, "clamp_fxp unsigned: -1.0 => 0.0");
    CHECK_FEQ(adc_clamp_fxp(&s,  1.0f),  1.0f, 1e-6f, "clamp_fxp unsigned:  1.0 unchanged");
    CHECK_FEQ(adc_clamp_fxp(&s,  2.0f),  2.0f, 1e-6f, "clamp_fxp unsigned:  2.0 unchanged");
    CHECK_FEQ(adc_clamp_fxp(&s,  3.75f), 3.75f,1e-6f, "clamp_fxp unsigned:  3.75 unchanged");
    CHECK_FEQ(adc_clamp_fxp(&s,  4.0f),  3.75f,1e-6f, "clamp_fxp unsigned:  4.0 => 3.75");

    /* (bitwidth=4, fracwidth=4, is_signed=True, ...) min=-0.5, max=0.4375 */
    s.total_bits = 4; s.frac_bits = 4; s.is_signed = 1;
    CHECK_FEQ(adc_clamp_fxp(&s, -0.6f),   -0.5f,   1e-6f, "clamp_fxp 4b4f s: -0.6 => -0.5");
    CHECK_FEQ(adc_clamp_fxp(&s, -0.5f),   -0.5f,   1e-6f, "clamp_fxp 4b4f s: -0.5 => -0.5");
    CHECK_FEQ(adc_clamp_fxp(&s, -0.25f),  -0.25f,  1e-6f, "clamp_fxp 4b4f s: -0.25 unchanged");
    CHECK_FEQ(adc_clamp_fxp(&s, -0.05f),  -0.05f,  1e-6f, "clamp_fxp 4b4f s: -0.05 unchanged");
    CHECK_FEQ(adc_clamp_fxp(&s,  0.45f),   0.4375f,1e-6f, "clamp_fxp 4b4f s:  0.45 => 0.4375");
    CHECK_FEQ(adc_clamp_fxp(&s,  0.6f),    0.4375f,1e-6f, "clamp_fxp 4b4f s:  0.6 => 0.4375");

    /* (bitwidth=4, fracwidth=4, is_signed=False, ...) min=0.0, max=0.9375 */
    s.is_signed = 0;
    CHECK_FEQ(adc_clamp_fxp(&s, -0.5f),    0.0f,   1e-6f, "clamp_fxp 4b4f u: -0.5 => 0.0");
    CHECK_FEQ(adc_clamp_fxp(&s, -0.25f),   0.0f,   1e-6f, "clamp_fxp 4b4f u: -0.25 => 0.0");
    CHECK_FEQ(adc_clamp_fxp(&s, -0.05f),   0.0f,   1e-6f, "clamp_fxp 4b4f u: -0.05 => 0.0");
    CHECK_FEQ(adc_clamp_fxp(&s,  0.4375f), 0.4375f,1e-6f, "clamp_fxp 4b4f u:  0.4375 unchanged");
    CHECK_FEQ(adc_clamp_fxp(&s,  1.4375f), 0.9375f,1e-6f, "clamp_fxp 4b4f u:  1.4375 => 0.9375");
}

/* ─── test_adc_quantize_float (fxp_to_int) ─────────────────────────────── */

static void test_fxp_to_int(void)
{
    SettingsADC s = g_sets;

    /* (bitwidth=6, fracwidth=2, is_signed=True,
     *  input=[-1.2343, 0.4434, 0.0032, -10.0, +10.0],
     *  expected=[-5, 2, 0, -32, 31]) */
    s.total_bits = 6; s.frac_bits = 2; s.is_signed = 1;
    CHECK_IEQ(adc_fxp_to_int(&s, -1.2343f), -5,  "fxp_to_int 6b s2: -1.2343 => -5");
    CHECK_IEQ(adc_fxp_to_int(&s,  0.4434f),  2,  "fxp_to_int 6b s2:  0.4434 =>  2");
    CHECK_IEQ(adc_fxp_to_int(&s,  0.0032f),  0,  "fxp_to_int 6b s2:  0.0032 =>  0");
    CHECK_IEQ(adc_fxp_to_int(&s, -10.0f),  -32,  "fxp_to_int 6b s2: -10.0  => -32 (clamped)");
    CHECK_IEQ(adc_fxp_to_int(&s,  10.0f),   31,  "fxp_to_int 6b s2: +10.0  =>  31 (clamped)");

    /* (bitwidth=6, fracwidth=2, is_signed=False,
     *  input=[-1.2343, 0.4434, 0.0032, -10.0, +20.0],
     *  expected=[0, 2, 0, 0, 63]) */
    s.is_signed = 0;
    CHECK_IEQ(adc_fxp_to_int(&s, -1.2343f),  0,  "fxp_to_int 6b u2: -1.2343 =>  0 (clamped)");
    CHECK_IEQ(adc_fxp_to_int(&s,  0.4434f),  2,  "fxp_to_int 6b u2:  0.4434 =>  2");
    CHECK_IEQ(adc_fxp_to_int(&s,  0.0032f),  0,  "fxp_to_int 6b u2:  0.0032 =>  0");
    CHECK_IEQ(adc_fxp_to_int(&s, -10.0f),    0,  "fxp_to_int 6b u2: -10.0   =>  0 (clamped)");
    CHECK_IEQ(adc_fxp_to_int(&s,  20.0f),   63,  "fxp_to_int 6b u2: +20.0   => 63 (clamped)");

    /* (bitwidth=8, fracwidth=4, is_signed=True,
     *  input=[-1.2343, 0.4434, 0.0032, -10.0, +10.0],
     *  expected=[-20, 7, 0, -128, 127]) */
    s.total_bits = 8; s.frac_bits = 4; s.is_signed = 1;
    CHECK_IEQ(adc_fxp_to_int(&s, -1.2343f), -20, "fxp_to_int 8b s4: -1.2343 => -20");
    CHECK_IEQ(adc_fxp_to_int(&s,  0.4434f),   7, "fxp_to_int 8b s4:  0.4434 =>   7");
    CHECK_IEQ(adc_fxp_to_int(&s,  0.0032f),   0, "fxp_to_int 8b s4:  0.0032 =>   0");
    CHECK_IEQ(adc_fxp_to_int(&s, -10.0f),  -128, "fxp_to_int 8b s4: -10.0   => -128 (clamped)");
    CHECK_IEQ(adc_fxp_to_int(&s,  10.0f),   127, "fxp_to_int 8b s4: +10.0   =>  127 (clamped)");

    /* (bitwidth=8, fracwidth=4, is_signed=False,
     *  input=[-1.2343, 0.4434, 0.0032, -10.0, +20.0],
     *  expected=[0, 7, 0, 0, 255]) */
    s.is_signed = 0;
    CHECK_IEQ(adc_fxp_to_int(&s, -1.2343f),   0, "fxp_to_int 8b u4: -1.2343 =>   0 (clamped)");
    CHECK_IEQ(adc_fxp_to_int(&s,  0.4434f),   7, "fxp_to_int 8b u4:  0.4434 =>   7");
    CHECK_IEQ(adc_fxp_to_int(&s,  0.0032f),   0, "fxp_to_int 8b u4:  0.0032 =>   0");
    CHECK_IEQ(adc_fxp_to_int(&s, -10.0f),     0, "fxp_to_int 8b u4: -10.0   =>   0 (clamped)");
    CHECK_IEQ(adc_fxp_to_int(&s,  20.0f),   255, "fxp_to_int 8b u4: +20.0   => 255 (clamped)");
}

/* ─── fxp_to_fxp ────────────────────────────────────────────────────────── */

static void test_fxp_to_fxp(void)
{
    SettingsADC s = g_sets;

    /* (bitwidth=6, fracwidth=2, is_signed=True,
     *  input=[-1.2343, 0.4434, 0.0032, -10.0, +10.0],
     *  expected=[-1.25, 0.5, 0.0, -8.0, 7.75]) */
    s.total_bits = 6; s.frac_bits = 2; s.is_signed = 1;
    CHECK_FEQ(adc_fxp_to_fxp(&s, -1.2343f), -1.25f, 1e-5f, "fxp_to_fxp: -1.2343 => -1.25");
    CHECK_FEQ(adc_fxp_to_fxp(&s,  0.4434f),  0.5f,  1e-5f, "fxp_to_fxp:  0.4434 =>  0.5");
    CHECK_FEQ(adc_fxp_to_fxp(&s,  0.0032f),  0.0f,  1e-5f, "fxp_to_fxp:  0.0032 =>  0.0");
    CHECK_FEQ(adc_fxp_to_fxp(&s, -10.0f),   -8.0f,  1e-5f, "fxp_to_fxp: -10.0   => -8.0");
    CHECK_FEQ(adc_fxp_to_fxp(&s,  10.0f),    7.75f, 1e-5f, "fxp_to_fxp: +10.0   =>  7.75");
}

/* ─── test_adc_quantize_integer (int_to_int) ───────────────────────────── */

static void test_int_to_int(void)
{
    SettingsADC s = g_sets;

    /* (bitwidth=6, fracwidth=2, is_signed=True,
     *  input=[-5, 2, 0, -40, 32], expected=[-5, 2, 0, -32, 31]) */
    s.total_bits = 6; s.frac_bits = 2; s.is_signed = 1;
    CHECK_IEQ(adc_int_to_int(&s,  -5),  -5, "int_to_int 6b s: -5 unchanged");
    CHECK_IEQ(adc_int_to_int(&s,   2),   2, "int_to_int 6b s:  2 unchanged");
    CHECK_IEQ(adc_int_to_int(&s,   0),   0, "int_to_int 6b s:  0 unchanged");
    CHECK_IEQ(adc_int_to_int(&s, -40), -32, "int_to_int 6b s: -40 => -32");
    CHECK_IEQ(adc_int_to_int(&s,  32),  31, "int_to_int 6b s:  32 =>  31");

    /* (bitwidth=6, fracwidth=2, is_signed=False,
     *  input=[0, 2, 0, 0, 63], expected=[0, 2, 0, 0, 63]) */
    s.is_signed = 0;
    CHECK_IEQ(adc_int_to_int(&s,   0),   0, "int_to_int 6b u:  0 unchanged");
    CHECK_IEQ(adc_int_to_int(&s,   2),   2, "int_to_int 6b u:  2 unchanged");
    CHECK_IEQ(adc_int_to_int(&s,  63),  63, "int_to_int 6b u: 63 unchanged");

    /* (bitwidth=8, fracwidth=4, is_signed=True,
     *  input=[-20, 7, 0, -140, 227], expected=[-20, 7, 0, -128, 127]) */
    s.total_bits = 8; s.frac_bits = 4; s.is_signed = 1;
    CHECK_IEQ(adc_int_to_int(&s,  -20),  -20, "int_to_int 8b s: -20 unchanged");
    CHECK_IEQ(adc_int_to_int(&s,    7),    7, "int_to_int 8b s:   7 unchanged");
    CHECK_IEQ(adc_int_to_int(&s,    0),    0, "int_to_int 8b s:   0 unchanged");
    CHECK_IEQ(adc_int_to_int(&s, -140), -128, "int_to_int 8b s: -140 => -128");
    CHECK_IEQ(adc_int_to_int(&s,  227),  127, "int_to_int 8b s:  227 =>  127");

    /* (bitwidth=8, fracwidth=4, is_signed=False,
     *  input=[0, 7, 0, -1, 355], expected=[0, 7, 0, 0, 255]) */
    s.is_signed = 0;
    CHECK_IEQ(adc_int_to_int(&s,    0),   0, "int_to_int 8b u:   0 unchanged");
    CHECK_IEQ(adc_int_to_int(&s,    7),   7, "int_to_int 8b u:   7 unchanged");
    CHECK_IEQ(adc_int_to_int(&s,   -1),   0, "int_to_int 8b u:  -1 => 0");
    CHECK_IEQ(adc_int_to_int(&s,  355), 255, "int_to_int 8b u: 355 => 255");
}

/* ─── test_adc_rescaling_voltage_fxp (voltage_to_fxp) ──────────────────── */

static void test_voltage_to_fxp(void)
{
    SettingsADC s = g_sets;

    /* (bitwidth=6, fracwidth=4, is_signed=True, vpos=0.5, vneg=-0.5)
     * expected: [-2.0, -1.75, -0.5, 0.0, 0.875, 1.8125, 2.0]
     * for input: [-0.55(→clamp), -0.434, -0.12, 0.0, 0.22, 0.45, 0.55(→clamp)] */
    s.total_bits = 6; s.frac_bits = 4; s.is_signed = 1;
    s.vneg = -0.5f; s.vpos = 0.5f;

    CHECK_FEQ(adc_voltage_to_fxp(&s, -0.55f), -2.0f,   1e-4f, "v_to_fxp: -0.55 => -2.0");
    CHECK_FEQ(adc_voltage_to_fxp(&s, -0.434f),-1.75f,  1e-4f, "v_to_fxp: -0.434 => -1.75");
    CHECK_FEQ(adc_voltage_to_fxp(&s, -0.12f), -0.5f,   1e-4f, "v_to_fxp: -0.12 => -0.5");
    CHECK_FEQ(adc_voltage_to_fxp(&s,  0.0f),   0.0f,   1e-4f, "v_to_fxp:  0.0  =>  0.0");
    CHECK_FEQ(adc_voltage_to_fxp(&s,  0.22f),  0.875f, 1e-4f, "v_to_fxp:  0.22 =>  0.875");
    CHECK_FEQ(adc_voltage_to_fxp(&s,  0.45f),  1.8125f,1e-4f, "v_to_fxp:  0.45 =>  1.8125");
    /* 0.55 → clamped to 0.5 → steps=64 → int=32, clamped to max_int=31 → fxp=31*0.0625=1.9375 */
    CHECK_FEQ(adc_voltage_to_fxp(&s,  0.55f),  1.9375f,1e-4f, "v_to_fxp:  0.55 => 1.9375 (clamped)");

    /* (bitwidth=8, fracwidth=7, is_signed=False, vpos=0.5, vneg=-0.5)
     * expected: [0.0, 0.1328125, 0.7578125, 1.0, 1.4375, 1.8984375, 2.0] */
    s.total_bits = 8; s.frac_bits = 7; s.is_signed = 0;
    CHECK_FEQ(adc_voltage_to_fxp(&s, -0.55f),  0.0f,       1e-5f, "v_to_fxp u: -0.55 =>  0.0");
    CHECK_FEQ(adc_voltage_to_fxp(&s, -0.434f), 0.1328125f, 1e-5f, "v_to_fxp u: -0.434 => 0.1328125");
    CHECK_FEQ(adc_voltage_to_fxp(&s, -0.12f),  0.7578125f, 1e-5f, "v_to_fxp u: -0.12 => 0.7578125");
    CHECK_FEQ(adc_voltage_to_fxp(&s,  0.0f),   1.0f,       1e-5f, "v_to_fxp u:  0.0  => 1.0");
    CHECK_FEQ(adc_voltage_to_fxp(&s,  0.22f),  1.4375f,    1e-5f, "v_to_fxp u:  0.22 => 1.4375");
    CHECK_FEQ(adc_voltage_to_fxp(&s,  0.45f),  1.8984375f, 1e-5f, "v_to_fxp u:  0.45 => 1.8984375");
    /* 0.55 → clamped to 0.5 → steps=256 → int=256, clamped to max_int=255 → fxp=255/128=1.9921875 */
    CHECK_FEQ(adc_voltage_to_fxp(&s,  0.55f),  1.9921875f, 1e-6f, "v_to_fxp u:  0.55 => 1.9921875 (clamped)");
}

/* ─── test_adc_quantize_from_voltage_to_int (voltage_to_int) ───────────── */

static void test_voltage_to_int(void)
{
    SettingsADC s = g_sets;

    /* (bitwidth=6, fracwidth=4, is_signed=True, vpos=0.5, vneg=-0.5)
     * expected: [-32, -28, -8, 0, 14, 29, 32] for
     * input: [-0.55(clamp), -0.434, -0.12, 0.0, 0.22, 0.45, 0.55(clamp)] */
    s.total_bits = 6; s.frac_bits = 4; s.is_signed = 1;
    s.vneg = -0.5f; s.vpos = 0.5f;
    CHECK_IEQ(adc_voltage_to_int(&s, -0.55f), -32, "v_to_int: -0.55 => -32");
    CHECK_IEQ(adc_voltage_to_int(&s, -0.12f),  -8, "v_to_int: -0.12 => -8");
    CHECK_IEQ(adc_voltage_to_int(&s,  0.0f),    0, "v_to_int:  0.0  =>  0");
    CHECK_IEQ(adc_voltage_to_int(&s,  0.22f),  14, "v_to_int:  0.22 =>  14");
    CHECK_IEQ(adc_voltage_to_int(&s,  0.55f),  31, "v_to_int:  0.55 =>  31 (clamped)");

    /* Sanity: symmetric range, 8-bit signed → 0V maps to 0 */
    s.total_bits = 8; s.frac_bits = 0; s.is_signed = 1;
    s.vneg = -1.0f; s.vpos = 1.0f;
    CHECK_IEQ(adc_voltage_to_int(&s, -1.0f), -128, "v_to_int 8b: -1.0 => -128");
    CHECK_IEQ(adc_voltage_to_int(&s,  0.0f),    0, "v_to_int 8b:  0.0 =>    0");
}

/* ─── test_adc_resampling ───────────────────────────────────────────────── */

static void test_resample_no_op(void)
{
    SettingsADC s = g_sets;
    s.srate_orig = 1.0f; s.srate_new = 1.0f;

    float in[4]  = { 1.0f, 2.0f, 3.0f, 4.0f };
    float out[8] = { 0 };

    /* same rate → passthrough */
    size_t n = adc_resample(&s, in, 4, out, 8);
    CHECK(n == 4, "resample no-op: length unchanged");
    CHECK_FEQ(out[0], 1.0f, 1e-6f, "resample no-op: out[0] == in[0]");
    CHECK_FEQ(out[3], 4.0f, 1e-6f, "resample no-op: out[3] == in[3]");

    /* srate_new=0 → passthrough */
    s.srate_new = 0.0f;
    n = adc_resample(&s, in, 4, out, 8);
    CHECK(n == 4, "resample srate_new=0: length unchanged");
}

static void test_resample_downsample_constant(void)
{
    /* constant input: any resampling method must return the same constant */
    SettingsADC s = g_sets;
    s.srate_orig = 100.0f; s.srate_new = 50.0f;

    float in[4]  = { 1.0f, 1.0f, 1.0f, 1.0f };
    float out[4] = { 0 };

    size_t n = adc_resample(&s, in, 4, out, 4);
    CHECK(n == 2, "resample 2x down: length halved");
    CHECK_FEQ(out[0], 1.0f, 1e-5f, "resample 2x down constant: out[0] == 1.0");
    CHECK_FEQ(out[1], 1.0f, 1e-5f, "resample 2x down constant: out[1] == 1.0");
}

static void test_resample_upsample_constant(void)
{
    SettingsADC s = g_sets;
    s.srate_orig = 100.0f; s.srate_new = 200.0f;

    float in[4]  = { 3.0f, 3.0f, 3.0f, 3.0f };
    float out[8] = { 0 };

    size_t n = adc_resample(&s, in, 4, out, 8);
    CHECK(n == 8, "resample 2x up: length doubled");
    for (size_t i = 0; i < n; i++)
        if (fabsf(out[i] - 3.0f) > 1e-5f) {
            CHECK(0, "resample 2x up constant: value != 3.0");
            return;
        }
    CHECK(1, "resample 2x up constant: all values == 3.0");
}

static void test_resample_linear_interp(void)
{
    /* Linear ramp: input [0.0, 1.0, 2.0, 3.0] at 100 Hz → downsample to 50 Hz.
     * ratio=2, out[0] at pos=0 → in[0]=0.0, out[1] at pos=2 → in[2]=2.0. */
    SettingsADC s = g_sets;
    s.srate_orig = 100.0f; s.srate_new = 50.0f;
    s.vneg = 0.0f; s.vpos = 0.0f;

    float in[4]  = { 0.0f, 1.0f, 2.0f, 3.0f };
    float out[4] = { 0 };

    adc_resample(&s, in, 4, out, 4);
    /* DC offset = in[0] = 0.0, so no DC effect here.
     * out[0]: pos=0.0 → in[0] = 0.0
     * out[1]: pos=2.0 → in[2] = 2.0 */
    CHECK_FEQ(out[0], 0.0f, 1e-5f, "resample linear: out[0] == 0.0");
    CHECK_FEQ(out[1], 2.0f, 1e-5f, "resample linear: out[1] == 2.0");
}

static void test_resample_out_len(void)
{
    SettingsADC s = g_sets;
    s.srate_orig = 100.0f; s.srate_new = 50.0f;
    CHECK(adc_resample_out_len(&s, 100) == 50,  "resample_out_len: 100 -> 50");

    s.srate_new = 200.0f;
    CHECK(adc_resample_out_len(&s, 100) == 200, "resample_out_len: 100 -> 200");

    s.srate_new = s.srate_orig;
    CHECK(adc_resample_out_len(&s, 100) == 100, "resample_out_len: same rate");
}

/* ─── test_adc_cutting_input_data ──────────────────────────────────────── */

static void test_cut_transient(void)
{
    SettingsADC s = g_sets;
    s.srate_orig = 5.0f; s.srate_new = 10.0f;

    float data[5] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    const float *ptr = NULL;
    size_t n;

    /* no cut: t_start >= t_stop → full array */
    n = adc_cut_transient(&s, data, 5, 0.0f, 0.0f, 1, &ptr);
    CHECK(n == 5 && ptr == data, "cut_transient: no cut => full array");

    /* (srate_orig=5, t=[0.0, 0.3]) → idx0=0, idx1=1 → 1 element */
    n = adc_cut_transient(&s, data, 5, 0.0f, 0.3f, 1, &ptr);
    CHECK(n == 1, "cut_transient orig: [0.0,0.3] => 1 element");
    CHECK(ptr == data, "cut_transient orig: pointer at start");

    /* (srate_orig=5, t=[0.2, 0.8]) → idx0=1, idx1=4 → 3 elements */
    n = adc_cut_transient(&s, data, 5, 0.2f, 0.8f, 1, &ptr);
    CHECK(n == 3, "cut_transient orig: [0.2,0.8] => 3 elements");
    CHECK(ptr == data + 1, "cut_transient orig: pointer offset=1");

    /* use_srate_orig=0: (srate_new=10, t=[0.0, 0.3]) → idx0=0, idx1=3 → 3 elements */
    n = adc_cut_transient(&s, data, 5, 0.0f, 0.3f, 0, &ptr);
    CHECK(n == 3, "cut_transient new: [0.0,0.3] => 3 elements");
}

/* ─── test_adc_cutting_input_labels ────────────────────────────────────── */

static void test_cut_labels(void)
{
    SettingsADC s = g_sets;
    s.srate_orig = 5.0f; s.srate_new = 10.0f;

    /* label_pos = [1,2,3,4,5]  (times: 0.2, 0.4, 0.6, 0.8, 1.0 at srate=5) */
    size_t lpos[5] = { 1, 2, 3, 4, 5 };
    size_t offset;
    size_t n;

    /* no cut */
    n = adc_cut_labels(&s, lpos, 5, 0.0f, 0.0f, 1, &offset);
    CHECK(n == 5 && offset == 0, "cut_labels: no cut => all labels");

    /* (srate_orig=5, t=[0.0, 0.3]) → first time >= 0.3 is 0.4 (pos=2) → [pos=1] → 1 label */
    n = adc_cut_labels(&s, lpos, 5, 0.0f, 0.3f, 1, &offset);
    CHECK(n == 1 && offset == 0, "cut_labels orig: [0.0,0.3] => 1 label at offset 0");

    /* (srate_orig=5, t=[0.3, 0.6]) → first >= 0.3 is pos=2(t=0.4); first >= 0.6 is pos=3(t=0.6) */
    n = adc_cut_labels(&s, lpos, 5, 0.3f, 0.6f, 1, &offset);
    CHECK(n == 1 && offset == 1, "cut_labels orig: [0.3,0.6] => 1 label at offset 1");
    CHECK(lpos[offset] == 2, "cut_labels orig: label_pos == 2");

    /* use_srate_orig=0: (srate_new=10, t=[0.0, 0.3]) → fpos1=3.0 → pos<3: 1,2 → 2 labels */
    n = adc_cut_labels(&s, lpos, 5, 0.0f, 0.3f, 0, &offset);
    CHECK(n == 2 && offset == 0, "cut_labels new: [0.0,0.3] => 2 labels");
}

/* ─── full pipeline: redefine_from_voltage (no resampling) ─────────────── */

static void test_redefine_from_voltage(void)
{
    /* same rate → resampling is a no-op; tests only the quantization path */
    SettingsADC s = g_sets;
    s.total_bits = 6; s.frac_bits = 4; s.is_signed = 1;
    s.srate_orig = 100.0f; s.srate_new = 100.0f;
    s.vneg = -0.5f; s.vpos = 0.5f;

    float   in[3]  = { -0.55f, 0.0f, 0.55f };
    float   tmp[6] = { 0 };
    int64_t out[3] = { 0 };

    size_t n = adc_redefine_from_voltage(&s, in, 3, tmp, 6, out);
    CHECK(n == 3,        "redefine_from_voltage: length correct");
    CHECK_IEQ(out[0], -32, "redefine_from_voltage: -0.55V => -32");
    CHECK_IEQ(out[1],   0, "redefine_from_voltage:  0.0V =>   0");
    CHECK_IEQ(out[2],  31, "redefine_from_voltage: +0.55V =>  31 (clamped)");
}

/* ─── full pipeline: redefine_from_fxp (no resampling) ─────────────────── */

static void test_redefine_from_fxp(void)
{
    SettingsADC s = g_sets;
    s.total_bits = 6; s.frac_bits = 2; s.is_signed = 1;
    s.srate_orig = 100.0f; s.srate_new = 100.0f;

    /* (bitwidth=6, fracwidth=2, is_signed=True,
     *  input=[-1.2343, 0.4434, 0.0032, -10.0, +10.0],
     *  expected=[-5, 2, 0, -32, 31]) */
    float   in[5]   = { -1.2343f, 0.4434f, 0.0032f, -10.0f, 10.0f };
    float   tmp[10] = { 0 };
    int64_t out[5]  = { 0 };

    size_t n = adc_redefine_from_fxp(&s, in, 5, tmp, 10, out);
    CHECK(n == 5,         "redefine_from_fxp: length correct");
    CHECK_IEQ(out[0], -5, "redefine_from_fxp: -1.2343 => -5");
    CHECK_IEQ(out[1],  2, "redefine_from_fxp:  0.4434 =>  2");
    CHECK_IEQ(out[2],  0, "redefine_from_fxp:  0.0032 =>  0");
    CHECK_IEQ(out[3],-32, "redefine_from_fxp: -10.0   => -32");
    CHECK_IEQ(out[4], 31, "redefine_from_fxp: +10.0   =>  31");
}

/* ─── full pipeline: redefine_from_int (no resampling) ─────────────────── */

static void test_redefine_from_int(void)
{
    SettingsADC s = g_sets;
    s.total_bits = 6; s.frac_bits = 2; s.is_signed = 1;
    s.srate_orig = 100.0f; s.srate_new = 100.0f;

    /* (bitwidth=6, fracwidth=2, is_signed=True,
     *  input=[-5, 2, 0, -40, 32], expected=[-5, 2, 0, -32, 31]) */
    int64_t in[5]  = { -5, 2, 0, -40, 32 };
    /* tmp must hold float(in) + resampled: 5 + 5 = 10 elements */
    float   tmp[10] = { 0 };
    int64_t out[5]  = { 0 };

    size_t n = adc_redefine_from_int(&s, in, 5, tmp, 10, out);
    CHECK(n == 5,         "redefine_from_int: length correct");
    CHECK_IEQ(out[0], -5, "redefine_from_int: -5  => -5");
    CHECK_IEQ(out[1],  2, "redefine_from_int:  2  =>  2");
    CHECK_IEQ(out[2],  0, "redefine_from_int:  0  =>  0");
    CHECK_IEQ(out[3],-32, "redefine_from_int: -40 => -32 (clamped)");
    CHECK_IEQ(out[4], 31, "redefine_from_int:  32 =>  31 (clamped)");
}

/* ─── main ──────────────────────────────────────────────────────────────── */

int main(void)
{
    setup(); test_vcm();
    setup(); test_lsb();
    setup(); test_clamp_analog();
    setup(); test_clamp_int();
    setup(); test_clamp_fxp();
    setup(); test_fxp_to_int();
    setup(); test_fxp_to_fxp();
    setup(); test_int_to_int();
    setup(); test_voltage_to_fxp();
    setup(); test_voltage_to_int();
    setup(); test_resample_no_op();
    setup(); test_resample_downsample_constant();
    setup(); test_resample_upsample_constant();
    setup(); test_resample_linear_interp();
    setup(); test_resample_out_len();
    setup(); test_cut_transient();
    setup(); test_cut_labels();
    setup(); test_redefine_from_voltage();
    setup(); test_redefine_from_fxp();
    setup(); test_redefine_from_int();

    printf("\n%d tests run:  %d passed,  %d failed\n",
           g_run, g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
