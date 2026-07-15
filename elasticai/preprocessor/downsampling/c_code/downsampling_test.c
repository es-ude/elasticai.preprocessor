// downsampling_test.c
// Compile:
//   gcc -std=c99 -Wall -Wextra -o downsampling_test \
//       downsampling_test.c downsampling.c -lm

#include "downsampling.h"

#include <math.h>
#include <stdio.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// minimal test harness 
static int g_run  = 0;
static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do {                         \
    g_run++;                                          \
    printf("%d. check:\t",g_run);                     \
    if (cond) {                                       \
        g_pass++;                                     \
        printf("%s passed.",msg);                     \
    } else {                                          \
        g_fail++;                                     \
        fprintf(stderr, "FAIL  %-52s  (%s:%d)\n",     \
                (msg), __FILE__, __LINE__);           \
    }                                                 \
    printf("\n");                                     \
} while (0)

#define CHECK_FEQ(a, b, eps, msg) \
    CHECK(fabsf((float)(a) - (float)(b)) <= (float)(eps), msg)

// signal parameters
#define FS          40000.0f
#define DSR         10
#define N_SAMPLES   40001       // linspace(0, 1, 40001, endpoint=True)
#define SENTINEL    1e38f       // value that cannot appear in filter output
#define CIC_STAGES  5

static SettingsDownSampling g_sets;
static float s_input[N_SAMPLES];
static float s_uout [N_SAMPLES + 1]; // +1: room for sentinel boundary check 
static float s_ref  [N_SAMPLES];     // reference buffer for equivalence tests

// FIR-equivalent CIC reference 

// Impulse response of a 5-stage CIC filter with DSR=10:
//  = 5-fold convolution of the boxcar [1,1,...,1] (length 10), divided by 10^5.
// Length = (DSR-1)*CIC_STAGES + 1 = 46.
// Computed with Python/numpy (float64) and embedded here as the drift-free
// mathematical ground truth. The do_cic() int64 implementation must match
// this reference within float32 output precision (~1e-6). 
static const double s_cic_h[46] = {
    1.0000000000000001e-05, 5.0000000000000002e-05, 1.4999999999999999e-04,
    3.5000000000000000e-04, 6.9999999999999999e-04, 1.2600000000000001e-03,
    2.0999999999999999e-03, 3.3000000000000000e-03, 4.9500000000000004e-03,
    7.1500000000000001e-03, 9.9600000000000001e-03, 1.3400000000000000e-02,
    1.7450000000000000e-02, 2.2050000000000000e-02, 2.7100000000000000e-02,
    3.2460000000000003e-02, 3.7950000000000000e-02, 4.3350000000000000e-02,
    4.8399999999999999e-02, 5.2800000000000000e-02, 5.6309999999999999e-02,
    5.8749999999999997e-02, 6.0000000000000000e-02, 6.0000000000000000e-02,
    5.8749999999999997e-02, 5.6309999999999999e-02, 5.2800000000000000e-02,
    4.8399999999999999e-02, 4.3350000000000000e-02, 3.7950000000000000e-02,
    3.2460000000000003e-02, 2.7100000000000000e-02, 2.2050000000000000e-02,
    1.7450000000000000e-02, 1.3400000000000000e-02, 9.9600000000000001e-03,
    7.1500000000000001e-03, 4.9500000000000004e-03, 3.3000000000000000e-03,
    2.0999999999999999e-03, 1.2600000000000001e-03, 6.9999999999999999e-04,
    3.5000000000000000e-04, 1.4999999999999999e-04, 5.0000000000000002e-05,
    1.0000000000000001e-05
};
#define CIC_H_LEN 46

// Compute the FIR-equivalent CIC output for output index out_idx.
// Negative input indices are treated as 0 (zero initial state).
static double fir_cic_ref(size_t out_idx, const float *inp)
{
    double acc = 0.0;
    for (int j = 0; j < CIC_H_LEN; j++) {
        int src = (int)(out_idx * (size_t)DSR) - j;
        if (src >= 0) acc += s_cic_h[j] * (double)inp[src];
    }
    return acc;
}

// mirrors Python: inp_samp(time) 
static void generate_input(void)
{
    const float freqs[2] = { 4.0f, 400.0f };
    for (size_t i = 0; i < N_SAMPLES; i++) {
        float t = (float)i / (float)(N_SAMPLES - 1); /* linspace(0,1,40001) */
        float z = 0.0f;
        for (int f = 0; f < 2; f++)
            z += sinf(2.0f * (float)M_PI * freqs[f] * t);
        s_input[i] = 0.75f * z / 2.0f;
    }
}

// mirrors Python: setUp()  –  called from main() before every test 
static void setup(void)
{
    g_sets.sampling_rate = FS;
    g_sets.dsr           = DSR;
    generate_input();
    for (size_t i = 0; i < N_SAMPLES + 1; i++) s_uout[i] = SENTINEL;
}

// tests 
static void test_output_sampling_rate(void)
{
    SettingsDownSampling s = { .sampling_rate = 10000.0f, .dsr = 4 };
    CHECK_FEQ(sampling_rate_out(&s), 2500.0f, 1e-3f,
              "sampling_rate_out: 10000 / 4 == 2500");
}

static void test_do_simple_size(void)
{
    // n = (40001 / 10) * 10 = 40000  →  n_blocks = 4000 
    const size_t expected = N_SAMPLES / (size_t)g_sets.dsr;   // 4000  
    bool result = do_simple(&g_sets, s_input, N_SAMPLES, s_uout);
    CHECK(result, "do_simple: return-val is true");
    CHECK(s_uout[expected - 1] != SENTINEL, "do_simple: last expected element is written");
    CHECK(s_uout[expected]     == SENTINEL, "do_simple: no write past expected size");
    CHECK(expected             == 4000u,    "do_simple: output count is 4000");
}

static void test_do_simple_no_uin(void)
{
    bool result = do_simple(&g_sets, NULL, N_SAMPLES, s_uout);
    CHECK(!result, "do_simple: no uin throws false");
}

static void test_do_simple_no_uout(void)
{
    bool result = do_simple(&g_sets, s_input, N_SAMPLES, NULL);
    CHECK(!result, "do_simple: no uout throws false");
}

static void test_do_simple_no_settings(void)
{
    bool result = do_simple(NULL, s_input, N_SAMPLES, s_uout);
    CHECK(!result, "do_simple: no settings throws false");
}

static void test_do_simple_dsr_is_zero(void)
{
    g_sets.dsr = 0;
    bool result = do_simple(&g_sets, s_input, N_SAMPLES, s_uout);
    CHECK(!result, "do_simple: dsr = 0 throws false");
}

static void test_do_simple_dsr_is_negative(void)
{
    g_sets.dsr = -10;
    bool result = do_simple(&g_sets, s_input, N_SAMPLES, s_uout);
    CHECK(!result, "do_simple: dsr is negative throws false");
}

static void test_do_simple_uin_len_is_zero(void)
{
    bool result = do_simple(&g_sets, s_input, 0, s_uout);
    CHECK(!result, "do_simple: uin_len = 0 throws false");
}

static void test_cic_first_value(void)
{
    /* input[0] = 0.0  (t=0: both sines are 0)
     * CIC fires at sample 0 (0 % DSR == 0); all integrators and combs
     * start at 0  →  z = 0  →  output[0] = 0.0 / gain = 0.0             */
    bool result = do_cic(&g_sets, s_input, N_SAMPLES, CIC_STAGES, s_uout);
    CHECK(result, "do_cic: return-val is true");
    CHECK_FEQ(s_uout[0], 0.0f, 1e-7f, "cic: output[0] == 0.0");
}

static void test_cic_no_settings(void)
{
    bool result = do_cic(NULL, s_input, N_SAMPLES, CIC_STAGES, s_uout);
    CHECK(!result, "do_cic: no settings throws false");
}

static void test_cic_no_uin(void)
{
    bool result = do_cic(&g_sets, NULL, N_SAMPLES, CIC_STAGES, s_uout);
    CHECK(!result, "do_cic: no uin throws false");
}

static void test_cic_no_uout(void)
{
    bool result = do_cic(&g_sets, s_input, N_SAMPLES, CIC_STAGES, NULL);
    CHECK(!result, "do_cic: no uout throws false");
}

static void test_cic_uin_len_is_zero(void)
{
    bool result = do_cic(&g_sets, s_input, 0, CIC_STAGES, s_uout);
    CHECK(!result, "do_cic: uin_len = 0 throws false");
}

static void test_cic_dsr_is_zero(void)
{
    g_sets.dsr = 0;
    bool result = do_cic(&g_sets, s_input, N_SAMPLES, CIC_STAGES, s_uout);
    CHECK(!result, "do_cic: dsr = 0 throws false");
}

static void test_cic_dsr_is_negative(void)
{
    g_sets.dsr = -10;
    bool result = do_cic(&g_sets, s_input, N_SAMPLES, CIC_STAGES, s_uout);
    CHECK(!result, "do_cic: dsr is negative throws false");
}

static void test_cic_size(void)
{
    /* Fires at 0, DSR, 2*DSR, …  →  count = (N_SAMPLES-1)/DSR + 1 = 4001 */
    const size_t expected = (N_SAMPLES - 1) / (size_t)g_sets.dsr + 1;  /* 4001 */
    do_cic(&g_sets, s_input, N_SAMPLES, CIC_STAGES, s_uout);
    CHECK(s_uout[expected - 1] != SENTINEL, "cic: last expected element is written");
    CHECK(s_uout[expected]     == SENTINEL, "cic: no write past expected size");
    CHECK(expected             == 4001u,    "cic: output count is 4001");
}

// compares every CIC output value against the FIR-equivalent reference.
// The FIR convolution uses double arithmetic and avoids the accumulation
// drift of float64 CIC integrators.  Tolerance 1e-5 (actual max diff ~7e-8). */
static void test_cic_all_values(void)
{
    do_cic(&g_sets, s_input, N_SAMPLES, CIC_STAGES, s_uout);
    const size_t n_out = (N_SAMPLES - 1) / (size_t)g_sets.dsr + 1;
    int    ok        = 1;
    size_t fail_idx  = 0;
    double fail_diff = 0.0;
    for (size_t i = 0; i < n_out; i++) {
        double ref  = fir_cic_ref(i, s_input);
        double diff = fabs(ref - (double)s_uout[i]);
        if (diff > 1e-5) { ok = 0; fail_idx = i; fail_diff = diff; break; }
    }
    if (!ok)
        printf("  (first failure at index %zu, diff=%.3e)\n",
               fail_idx, fail_diff);
    CHECK(ok, "cic: all output values match FIR reference (tol 1e-5)");
}

static void test_polyphase_one_no_uin(void)
{
    bool result = do_decimation_polyphase_order_one(NULL, N_SAMPLES, s_uout);
    CHECK(!result, "polyphase_one: no uin throws false");
}

static void test_polyphase_one_no_uout(void)
{
    bool result = do_decimation_polyphase_order_one(s_input, N_SAMPLES, NULL);
    CHECK(!result, "polyphase_one: no uout throws false");
}

static void test_polyphase_one_uin_len_is_zero(void)
{
    bool result = do_decimation_polyphase_order_one(s_input, 0, s_uout);
    CHECK(!result, "polyphase_one: uin_len = 0 throws false");
}

static void test_polyphase_one_first_value(void)
{
    /* idx=1 (first odd index):
     *   output[0] = input[1] + last_hs   where last_hs initialises to 0
     *             = input[1] + 0.0  =  input[1]
     * Python golden value[0]: 0.023782064257008607                         */
    bool result = do_decimation_polyphase_order_one(s_input, N_SAMPLES, s_uout);
    CHECK(result, "poyphase_one: return-val is true");
    CHECK_FEQ(s_uout[0], s_input[1], 1e-6f,
              "polyphase_one: output[0] == input[1]  (last_hs init = 0)");
    CHECK_FEQ(s_uout[0], 0.023782f, 1e-4f,
              "polyphase_one: output[0] matches Python golden value");
}

static void test_polyphase_one_size(void)
{
    /* Odd indices in [0..40000]: count = 40001 / 2 = 20000  (integer div) */
    const size_t expected = N_SAMPLES / 2;   /* 20000 */
    do_decimation_polyphase_order_one(s_input, N_SAMPLES, s_uout);
    CHECK(s_uout[expected - 1] != SENTINEL, "polyphase_one: last expected element is written");
    CHECK(s_uout[expected]     == SENTINEL, "polyphase_one: no write past expected size");
    CHECK(expected             == 20000u,   "polyphase_one: output count is 20000");
}

static void test_polyphase_one_finite(void)
{
    do_decimation_polyphase_order_one(s_input, N_SAMPLES, s_uout);
    const size_t n = N_SAMPLES / 2;
    bool ok = true;
    for (size_t i = 0; i < n; i++)
        if (!isfinite(s_uout[i])) { ok = false; break; }
    CHECK(ok, "polyphase_one: all output values are finite (not NaN / Inf)");
}

static void test_polyphase_two_no_uin(void)
{
    bool result = do_decimation_polyphase_order_two(NULL, N_SAMPLES, s_uout);
    CHECK(!result, "polyphase_two: no uin throws false");
}

static void test_polyphase_two_no_uout(void)
{
    bool result = do_decimation_polyphase_order_two(s_input, N_SAMPLES, NULL);
    CHECK(!result, "polyphase_two: no uout throws false");
}

static void test_polyphase_two_uin_len_is_zero(void)
{
    bool result = do_decimation_polyphase_order_two(s_input, 0, s_uout);
    CHECK(!result, "polyphase_two: uin_len = 0 throws false");
}

static void test_polyphase_two_first_value(void)
{
    /* idx=1 (odd):
     *   output[0] = input[1] + 2*last_even(=input[0]=0) + last_even_prev(=0)
     *             = input[1]
     * Both extra terms vanish: input[0]=0 (t=0) and last_even_prev inits to 0.
     * Same first element as polyphase_one (confirmed by Python).            */
    bool result = do_decimation_polyphase_order_two(s_input, N_SAMPLES, s_uout);
    CHECK(result, "polyphase_two: return-val is true");
    CHECK_FEQ(s_uout[0], s_input[1], 1e-6f,
              "polyphase_two: output[0] == input[1]  (states init = 0)");
    CHECK_FEQ(s_uout[0], 0.023782f, 1e-4f,
              "polyphase_two: output[0] matches Python golden value");
}

static void test_polyphase_two_size(void)
{
    const size_t expected = N_SAMPLES / 2;   /* 20000 */
    do_decimation_polyphase_order_two(s_input, N_SAMPLES, s_uout);
    CHECK(s_uout[expected - 1] != SENTINEL, "polyphase_two: last expected element is written");
    CHECK(s_uout[expected]     == SENTINEL, "polyphase_two: no write past expected size");
    CHECK(expected             == 20000u,   "polyphase_two: output count is 20000");
}

static void test_polyphase_two_finite(void)
{
    do_decimation_polyphase_order_two(s_input, N_SAMPLES, s_uout);
    const size_t n = N_SAMPLES / 2;
    bool ok = true;
    for (size_t i = 0; i < n; i++)
        if (!isfinite(s_uout[i])) { ok = false; break; }
    CHECK(ok, "polyphase_two: all output values are finite (not NaN / Inf)");
}

static void test_polyphase_no_settings(void)
{
    bool result = do_decimation_polyphase(NULL, s_input, N_SAMPLES, false, s_uout);
    CHECK(!result, "polyphase no settings: throws false");
}

static void test_polyphase_no_uin(void)
{
    bool result = do_decimation_polyphase(&g_sets, NULL, N_SAMPLES, false, s_uout);
    CHECK(!result, "polyphase no uin: throws false");
}

static void test_polyphase_no_uout(void)
{
    bool result = do_decimation_polyphase(&g_sets, s_input, N_SAMPLES, false, NULL);
    CHECK(!result, "polyphase no uout: throws false");
}

static void test_polyphase_uin_len_is_zero(void)
{
    bool result = do_decimation_polyphase(&g_sets, s_input, 0, false, s_uout);
    CHECK(!result, "polyphase uin_len=0: throws false");
}

static void test_polyphase_dsr_is_zero(void)
{
    SettingsDownSampling s = { .sampling_rate = FS, .dsr = 0 };
    bool result = do_decimation_polyphase(&s, s_input, N_SAMPLES, false, s_uout);
    CHECK(!result, "polyphase dsr=0: throws false");
}

static void test_polyphase_dsr_is_one(void)
{
    SettingsDownSampling s = { .sampling_rate = FS, .dsr = 1 };
    bool result = do_decimation_polyphase(&s, s_input, N_SAMPLES, false, s_uout);
    CHECK(!result, "polyphase dsr=1: throws false");
}

static void test_polyphase_invalid_dsr(void)
{
    SettingsDownSampling s = { .sampling_rate = FS, .dsr = 3 };
    bool result = do_decimation_polyphase(&s, s_input, N_SAMPLES, 0, s_uout);
    CHECK(!result,              "polyphase dsr=3: throws false");
    CHECK(s_uout[0] == SENTINEL, "polyphase dsr=3: no output written (not power of two)");
}

static void test_polyphase_dsr2_order_two(void)
{
    SettingsDownSampling s = { .sampling_rate = FS, .dsr = 2 };
    do_decimation_polyphase_order_two(s_input, N_SAMPLES, s_ref);
    do_decimation_polyphase(&s, s_input, N_SAMPLES, 0, s_uout);
    CHECK_FEQ(s_uout[0],   s_ref[0],   1e-6f, "polyphase dsr=2 order_two: output[0] matches");
    CHECK_FEQ(s_uout[100], s_ref[100], 1e-6f, "polyphase dsr=2 order_two: output[100] matches");
    CHECK_FEQ(s_uout[999], s_ref[999], 1e-6f, "polyphase dsr=2 order_two: output[999] matches");
}

static void test_polyphase_dsr2_order_one(void)
{
    SettingsDownSampling s = { .sampling_rate = FS, .dsr = 2 };
    do_decimation_polyphase_order_one(s_input, N_SAMPLES, s_ref);
    do_decimation_polyphase(&s, s_input, N_SAMPLES, 1, s_uout);
    CHECK_FEQ(s_uout[0],   s_ref[0],   1e-6f, "polyphase dsr=2 order_one: output[0] matches");
    CHECK_FEQ(s_uout[100], s_ref[100], 1e-6f, "polyphase dsr=2 order_one: output[100] matches");
    CHECK_FEQ(s_uout[999], s_ref[999], 1e-6f, "polyphase dsr=2 order_one: output[999] matches");
}

static void test_polyphase_dsr4_size(void)
{
    SettingsDownSampling s = { .sampling_rate = FS, .dsr = 4 };
    const size_t expected = N_SAMPLES / 4;   /* 10000 */
    do_decimation_polyphase(&s, s_input, N_SAMPLES, 0, s_uout);
    CHECK(s_uout[expected - 1] != SENTINEL, "polyphase dsr=4: last expected element written");
    CHECK(s_uout[expected]     == SENTINEL, "polyphase dsr=4: no write past expected size");
}

// main 
int main(void)
{
    setup(); test_output_sampling_rate();
    setup(); test_do_simple_size();
    setup(); test_do_simple_no_uin();
    setup(); test_do_simple_no_uout();
    setup(); test_do_simple_no_settings();
    setup(); test_do_simple_dsr_is_zero();
    setup(); test_do_simple_dsr_is_negative();
    setup(); test_do_simple_uin_len_is_zero();
    setup(); test_cic_first_value();
    setup(); test_cic_no_settings();
    setup(); test_cic_no_uin();
    setup(); test_cic_no_uout();
    setup(); test_cic_uin_len_is_zero();
    setup(); test_cic_dsr_is_zero();
    setup(); test_cic_dsr_is_negative();
    setup(); test_cic_size();
    setup(); test_cic_all_values();
    setup(); test_polyphase_one_no_uin();
    setup(); test_polyphase_one_no_uout();
    setup(); test_polyphase_one_uin_len_is_zero();
    setup(); test_polyphase_one_first_value();
    setup(); test_polyphase_one_size();
    setup(); test_polyphase_one_finite();
    setup(); test_polyphase_two_no_uin();
    setup(); test_polyphase_two_no_uout();
    setup(); test_polyphase_two_uin_len_is_zero();
    setup(); test_polyphase_two_first_value();
    setup(); test_polyphase_two_size();
    setup(); test_polyphase_two_finite();
    setup(); test_polyphase_no_settings();
    setup(); test_polyphase_no_uin();
    setup(); test_polyphase_no_uout();
    setup(); test_polyphase_uin_len_is_zero();
    setup(); test_polyphase_dsr_is_zero();
    setup(); test_polyphase_dsr_is_one();
    setup(); test_polyphase_invalid_dsr();
    setup(); test_polyphase_dsr2_order_two();
    setup(); test_polyphase_dsr2_order_one();
    setup(); test_polyphase_dsr4_size();

    printf("\n%d tests run:  %d passed,  %d failed\n",
           g_run, g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
