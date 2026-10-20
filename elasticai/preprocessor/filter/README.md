# Filtering

Preprocessing module that applies a **digital IIR or FIR filter** to a transient data stream and emulates the behaviour of the corresponding hardware implementation. The filter coefficients are designed automatically with `scipy.signal` from the settings. The module offers floating-point filtering, a bit-accurate quantized filtering, coefficient export for Verilog and the generation of hardware designs for MCU/PC (C) and FPGA (Verilog).

## How it works

When the module is created, the filter coefficients `b` (numerator) and `a` (denominator) are calculated from `FilterSettings`. `forward()` applies them with `scipy.signal.lfilter`.

### Filter design

| `method` | Design                                                                                                      | Meaning of `n_order`      |
| -------- | ----------------------------------------------------------------------------------------------------------- | ------------------------- |
| `IIR`    | `scipy.signal.iirfilter` with structure `f_type` and type `b_type`. Notch and allpass are designed separately. | Filter order (for `Notch`: quality factor `Q`) |
| `FIR`    | `scipy.signal.firwin` (window method). Notch uses `firwin2`, allpass is a pure delay line.                  | **Number of taps**        |

Available filter types (`b_type`) and the expected content of `f_filt`:

| `b_type`   | IIR                                                    | FIR                                                       |
| ---------- | ------------------------------------------------------ | --------------------------------------------------------- |
| `Lowpass`  | `[f_c]`                                                | `[f_c]`                                                   |
| `Highpass` | `[f_c]`                                                | `[f_c]`                                                   |
| `Bandpass` | `[f_low, f_high]`                                      | `[f_low, f_high]`                                         |
| `Bandstop` | `[f_low, f_high]`                                      | `[f_low, f_high]`                                         |
| `Notch`    | `[f_notch]` (`Q = n_order`)                            | `[f_notch, bandwidth]`                                    |
| `Allpass`  | `n_order=1`: `[f_b]`, `n_order=2`: `[f_b, bandwidth]`  | `[f_delay]`, delay line with `int(sampling_rate / f_delay)` taps |

All frequencies are given in Hz.

Special cases:

- **FIR, `n_order=1`, `f_filt[0] = sampling_rate / 2`:** simple two-tap average filter with coefficients `[0.5, 0.5]`. It has its own, resource-saving hardware implementation.
- **IIR filter structures (`f_type`):** `butter`, `cheby1`, `cheby2`, `ellip`, `bessel`. The settings contain no ripple parameters (`rp`/`rs`), which `scipy` requires for Chebyshev and elliptic filters. Without them, `butter` and `bessel` can be used directly.

## Settings

Configuration is done with `FilterSettings`:

| Attribute       | Type                      | Description                                                       |
| --------------- | ------------------------- | ----------------------------------------------------------------- |
| `method`        | `FilterTargets`           | Filter algorithm (`IIR`, `FIR`)                                   |
| `sampling_rate` | `float`                   | Sampling rate of the data stream [Hz]                             |
| `n_order`       | `int`                     | Filter order (IIR) or number of taps (FIR)                        |
| `f_filt`        | `list`                    | Filter frequencies [Hz], see table above                          |
| `f_type`        | `FilterFTypes`            | Filter structure (`Butter`, `Cheby1`, `Cheby2`, `Ellip`, `Bessel`)|
| `b_type`        | `FilterBTypes`            | Filter type (`Lowpass`, `Highpass`, `Bandpass`, `Bandstop`, `Notch`, `Allpass`) |

`method`, `f_type` and `b_type` can also be given as lowercase strings (e.g. `"iir"`, `"butter"`, `"lowpass"`).

A default configuration (IIR, Butterworth bandpass, order 2, `[0.1, 100]` Hz, 300 Hz sampling rate) is available as `FilterSettingsDefault`.

## Usage

```python
import numpy as np

from elasticai.preprocessor.filtering import (
    Filtering,
    FilterSettings,
    FilterTargets,
    FilterFTypes,
    FilterBTypes,
)

settings = FilterSettings(
    method=FilterTargets.IIR,
    sampling_rate=1000.0,
    n_order=2,
    f_filt=[50.0],
    f_type=FilterFTypes.Butter,
    b_type=FilterBTypes.Lowpass,
)

dut = Filtering(settings=settings)

x = np.random.randn(1000)
y = dut.forward(x)                                    # floating-point filtering
coeffs = dut.get_coeffs()                             # FilterCoeffs(b=[...], a=[...])
quant, error = dut.get_coeffs_quantized(bit_size=12)  # quantized coefficients and quantization error
```
An example for applying input data (top) and returning the filtered data (bottom):
<p>
<img src="elasticai/preprocessor/filter/docu/example.svg" width="600">
</p>
Notes:

- Called with a `DataTransient` (`__call__`), the module returns a `DataTransient` with the filtered data and the **unchanged sampling rate**. If the sampling rate of the incoming data differs from the settings, the settings are updated and the filter is redesigned.

### Quantized filtering

```python
y_q = dut.filt_quantized(xin=x, total_bitwidth=12, fraction_width=11, is_signed=True)
```

This emulates the hardware behaviour: the coefficients are quantized, the input and the output are quantized to the given fixed-point format, and an offset is subtracted that matches the hardware implementation (3 LSB for IIR, 1 LSB for negative samples for FIR, none for the FIR allpass).

Quantization of the coefficients (`get_coeffs_quantized`, signed):

| Method | Fractional bits   | Coefficient range |
| ------ | ----------------- | ----------------- |
| `FIR`  | `bit_size - 1`    | `[-1, 1)`         |
| `IIR`  | `bit_size - 2`    | `[-2, 2)`         |

`filt_quantized()` replaces the stored coefficients by the quantized ones (except for the FIR allpass). Subsequent calls of `forward()` therefore use the quantized coefficients.

### Coefficient export

`get_coeffs_verilog_string(bitwidth, only_half_fir=False)` returns the quantized coefficients as hex string array for Verilog.

- **FIR:** the coefficients `b` (with `only_half_fir=True` only the first half of the symmetric filter)
- **IIR:** the coefficients `b` followed by the negated coefficients `-a[1:]`

### Plots

| Method                                                    | Description                                      |
| --------------------------------------------------------- | ------------------------------------------------ |
| `plot_freq_response(num_points, show_plot, path2save)`    | Gain and phase of the filter, optionally saved   |
| `plot_grp_delay(num_points, show_plot)`                   | Group delay of the filter                        |

The responses are evaluated with `scipy.signal.freqs` (analog domain).

### Hardware design generation

`create_design(id, target, path2save, total_bitwidth, frac_bitwidth=0, is_signed=True)` exports the filter as hardware block. Supported targets (case-insensitive): `"mcu"`, `"pc"` (C code) and `"fpga"` (Verilog). `total_bitwidth` must be between 2 and 32.

**C code (`"mcu"`, `"pc"`)**

| Filter                                              | Implementation          |
| --------------------------------------------------- | ----------------------- |
| IIR                                                 | `build_filter_iir`      |
| FIR lowpass with `f_filt[0] = sampling_rate / 2`    | `build_filter_simple`   |
| FIR allpass                                         | `build_filter_delay`    |
| other FIR                                           | `build_filter_fir` (optimized half-coefficient version if the number of taps is odd) |

**Verilog (`"fpga"`)**

| Filter                                              | Verilog module | Remark                                       |
| --------------------------------------------------- | -------------- | -------------------------------------------- |
| IIR                                                 | `biquad_df1`   | only `n_order = 2`, direct form 1             |
| FIR with `f_filt[0] = sampling_rate / 2`            | `fir_low`      | two-tap average                              |
| FIR allpass                                         | `fir_delay`    | delay line                                   |
| FIR, odd number of taps                             | `fir_half`     | exploits the symmetry of the coefficients     |
| FIR, even number of taps                            | `fir_full`     | all coefficients                              |

Requirements and behaviour:

- Unsupported targets raise a `ValueError`, for the Verilog design also IIR orders other than 2 and `SimpleLowpass`.
- `is_signed` is only used for the C designs. `frac_bitwidth` is not used.
- The Verilog designs use one multiplier (`NUM_MULT = 1`), implemented with DSP blocks.