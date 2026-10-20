# ConversionResampler (ADC)

Preprocessing module that emulates an **Analog-Digital Converter (ADC)** on pre-recorded transient data. It optionally **resamples** the input signal to a new sampling rate and then **quantizes** it to a fixed-point (FxP) or integer representation. The result can be exported as a hardware design (C for MCU/PC, Verilog for FPGA) that replays the data stream.

## How it works

`ConversionResampler` processes data in two stages inside `forward()`:

1. **Resampling** (`_do_resample`)
   - The ratio `srate_new / srate_orig` is converted into a fraction `p/q`.
   - The signal is resampled with `scipy.signal.resample_poly` (Hamming window, `padtype="mean"`).
   - To avoid edge artifacts, the first sample is subtracted before and added after resampling (DC offset removal).
   - Resampling is skipped if `srate_new == 0.0`, `srate_new == srate_orig` or the data contains only one sample.

2. **Quantization**, depending on `method`:

   | `method`                                  | Input data                          | Behaviour                                                                                                                       |
   | ----------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
   | `ConversionResamplerTargets.FromVoltage`  | Analog values in `[vneg, vpos]`     | Clamps to the voltage range, maps to ADC codes with the step size `lsb = (vpos - vneg) / 2**total_bits` and converts to FxP/int |
   | `ConversionResamplerTargets.FromFxp`      | Fixed-point (rational) values       | Clamps to the representable FxP range and re-quantizes to the new bit configuration                                              |
   | `ConversionResamplerTargets.FromInteger`  | Integer codes                       | Clamps to the integer range, converts to rational values and re-quantizes                                                        |

   The output representation is selected by `is_int_output`:
   - `True`: integer codes (`int8/16/32/64` or `uint8/16/32/64`, chosen from `total_bits` and `is_signed`)
   - `False`: fixed-point values as rational numbers (`float32`)

   For `FromVoltage`, `vneg` is mapped to the lowest code and `vpos` to the highest code of the configured FxP format. E.g. for a signed `FxP(12, 11)` the voltage range `[-0.1 V, 0.1 V]` is mapped to `[-1.0, 1.0)`.

## Settings

Configuration is done with `ConversionResamplerSettings`:

| Attribute       | Type                         | Description                                                            |
| --------------- | ---------------------------- | ---------------------------------------------------------------------- |
| `method`        | `ConversionResamplerTargets` | Type of the input data (`FromVoltage`, `FromFxp`, `FromInteger`)       |
| `total_bits`    | `int`                        | Total number of bits of the output format                              |
| `frac_bits`     | `int`                        | Number of fractional bits (`0` = integer only)                         |
| `is_signed`     | `bool`                       | Signed (`True`) or unsigned (`False`) conversion                       |
| `is_int_output` | `bool`                       | Output as integer (`True`) or fixed-point (`False`)                    |
| `srate_orig`    | `float`                      | Sampling rate of the input data stream [Hz]                            |
| `srate_new`     | `float`                      | Sampling rate of the output data stream [Hz] (`0.0` = no resampling)   |
| `vneg`          | `float`                      | Minimum (negative) input voltage (default `-1.0`)                      |
| `vpos`          | `float`                      | Maximum (positive) input voltage (default `1.0`)                       |

Derived properties:

- `vcm`: common-mode voltage `(vpos + vneg) / 2`
- `lsb`: voltage of the least significant bit `(vpos - vneg) / 2**total_bits`

A ready-to-use unsigned 8-bit configuration for a 0 … 1.8 V range is available as `ConversionResamplerSettingsDefault`.

## Usage

```python
from elasticai.preprocessor import PreprocessingSequential
from elasticai.preprocessor.adc import (
    ConversionResampler,
    ConversionResamplerSettings,
    ConversionResamplerTargets,
)

settings = ConversionResamplerSettings(
    method=ConversionResamplerTargets.FromVoltage,
    total_bits=12,
    frac_bits=11,
    is_signed=True,
    is_int_output=False,
    srate_orig=1000.0,
    srate_new=2000.0,
    vneg=-0.1,
    vpos=0.1,
)

dut = PreprocessingSequential(ConversionResampler(settings=settings))
out = dut(x=data, fs=sampling_rate)   # out.data, out.sampling_rate
```
An example for applying input data (top) and returning the conversion data (bottom):
<p>
<img src="elasticai/preprocessor/adc/docu/example_adc.svg" width="600">
</p>
Notes:

- When called with a `DataTransient` (`__call__`), `srate_orig` is overwritten by the sampling rate of the incoming data. The output carries `srate_new` as its sampling rate.
- `forward(x)` can also be used directly on a `numpy` array (shape `(num_samples,)`).

### Additional helpers

| Method                                                                   | Description                                                                         |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `do_cut_transient(data, t_range_sec, use_srate_orig=True)`               | Cuts the transient signal to `[start, stop]` in seconds (empty list = no cut)       |
| `do_cut_labels(label_id, label_pos, t_range_sec, use_srate_orig=True)`   | Cuts event labels and their positions to the same time range                        |
| `add_trigger_list(trgg)`                                                 | Adds trigger positions that are included in the generated hardware design           |

Both cut functions raise a `ValueError` if `srate_orig` or `srate_new` is `0.0`, if the time range has a wrong order or if it does not have exactly 0 or 2 entries.

### Hardware design generation

`create_design(id, target, path2save, ...)` exports the quantized data as a replayer:

| `target`        | Output                                                                                |
| --------------- | ------------------------------------------------------------------------------------- |
| `"mcu"`, `"pc"` | C code (`c_compile.build_replayer` / `build_replayer_with_trigger` if triggers exist)  |
| `"fpga"`        | Verilog replayer plus `replayer_<id>_data.mem` (and `replayer_<id>_trgg.mem` with triggers) |

Requirements and behaviour:

- `forward()` must have been called before, otherwise a `ValueError` is raised.
- If the output was fixed-point (`is_int_output=False`), the data is automatically converted to integer codes for the export.
- When using `PreprocessingSequential`, the call `dut.create_design(settings=CreateSequentialSettings(...))` forwards this to all modules.

# Rescaler

Preprocessing module that **redefines the sample values of a digital data stream** to a desired output bit width and number format. It expects integer data (e.g. the output of an ADC) and maps it either by clamping to a fixed-point range or by scaling to an integer range (symmetric or asymmetric). The sampling rate is not changed.

Typical use case: the ADC delivers e.g. 12-bit codes, but the following model or hardware block works with 8 bits.

## How it works

`Rescaler.forward()` requires **integer input data**. Float arrays raise an `AttributeError`. Depending on `method`, one of the following steps is applied:

| `method`                | Behaviour                                                                                                         | Scaling parameters |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------ |
| `RescalerTargets.FxP`   | Clamps the integer codes to the representable range of the output format. Values inside the range stay unchanged. | not used           |
| `RescalerTargets.IntSym`| Symmetric scaling around zero: `x_q = round(x / scale)`                                                           | `zero_point = 0`   |
| `RescalerTargets.IntAsym`| Asymmetric scaling to the full integer range: `x_q = round(x / scale) + zero_point`                              | `scale`, `zero_point` |

Details of the scaling methods (`qmin`/`qmax` = integer limits of the output format):

- **`IntSym`**
  - `scale = max(|x|) / qmax`
  - Output range: `[-qmax, qmax]` for signed output, `[0, qmax]` for unsigned output
- **`IntAsym`**
  - `scale = (x_max - x_min) / (qmax - qmin)`
  - `zero_point = round(qmin - x_min / scale)`
  - Output range: `[qmin, qmax]`

The output is always an integer array. Its dtype is chosen from `total_bits` and `is_signed` (`int8/16/32/64` or `uint8/16/32/64`).

### Parameter update

By default (`update_params=True`) `scale` and `zero_point` are **recalculated from the data of every call** of `forward()`. To use fixed parameters (e.g. determined on a training data set, or to get the same mapping for all recordings), set them with `apply_params(scaling, bias)`. This disables the automatic update.

## Settings

Configuration is done with `RescalerSettings`:

| Attribute    | Type             | Description                                                            |
| ------------ | ---------------- | ---------------------------------------------------------------------- |
| `method`     | `RescalerTargets`| Rescaling method (`FxP`, `IntSym`, `IntAsym`)                          |
| `total_bits` | `int`            | Total bit width of the output data                                     |
| `frac_bits`  | `int`            | Number of fractional bits of the output format                         |
| `is_signed`  | `bool`           | Signed (`True`) or unsigned (`False`) output                           |

A default configuration (`FxP`, unsigned, 8 bits, 2 fractional bits) is available as `RescalerSettingsDefault`.

## Usage

```python
import numpy as np

from elasticai.preprocessor.rescaler import (
    Rescaler,
    RescalerSettings,
    RescalerTargets,
)

settings = RescalerSettings(
    method=RescalerTargets.IntSym,
    total_bits=8,
    frac_bits=0,
    is_signed=True,
)

dut = Rescaler(settings=settings)
x = np.array([-2048, -1024, 0, 1024, 2047], dtype=np.int16)   # e.g. 12-bit ADC codes
y = dut(x=x, fs=1.0).data                                     # int8 values in [-127, 127]
```
An example for applying input data (top) and returning the rescaling data (bottom):
<p>
<img src="elasticai/preprocessor/adc/docu/example_rescaler.svg" width="600">
</p>

Using fixed parameters instead of a data-dependent calculation:
```python
dut = Rescaler(settings=settings)
dut.apply_params(scaling=16.0, bias=0.0)
y = dut(x=x, fs=1.0)
```

Notes:

- Called with a `DataTransient` (`__call__`), the module returns a new `DataTransient` with rescaled data and the **unchanged sampling rate**.
- The module can be placed behind an ADC module (`ConversionResampler` with `is_int_output=True`) to adapt the bit width of the data stream.

### Hardware design generation

`create_design(id, target, path2save, total_bitwidth, frac_bitwidth, is_signed)` exports the rescaler as hardware block:

| `target`                                                    | Output                                                             |
| ----------------------------------------------------------- | ------------------------------------------------------------------ |
| `BuildPlatformTargets.Workstation`, `BuildPlatformTargets.MCU` | C code (`c_compile.build_rescaler`) including `scaling_factor` and `zero_point` |
| FPGA                                                        | not implemented yet (`NotImplementedError`)                        |

Requirements and behaviour:

- `total_bitwidth` is the bit width of the **input data** of the block.
- Call `forward()` before `create_design()`, so that `scale` and `zero_point` are set (either calculated from the data or via `apply_params`).
- `target` can be given as string or as `BuildPlatformTargets`.