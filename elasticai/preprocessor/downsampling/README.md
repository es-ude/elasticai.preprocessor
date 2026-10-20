# DownSampling

Preprocessing module that **reduces the sampling rate** of a transient data stream by an integer downsampling ratio `dsr`. Four methods are available (subsampling, block mean, CIC filter, polyphase decimation). The module can also be exported as a hardware design for MCU/PC (C), FPGA (Verilog) and ASIC (Verilog).

The output sampling rate is `sampling_rate_out = sampling_rate / dsr`.

## How it works

`DownSampling.forward()` applies the method selected by `method`:

| `method`                         | Behaviour                                                                                                                                | Output length (N input samples) |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `DownSamplingTargets.Subsampling`| Takes every `dsr`-th sample (no filtering)                                                                                               | `ceil(N / dsr)`                 |
| `DownSamplingTargets.Simple`     | Mean over non-overlapping blocks of `dsr` samples. Remaining samples at the end that do not fill a block are dropped.                    | `floor(N / dsr)`                |
| `DownSamplingTargets.CIC`        | Cascaded integrator-comb filter with `num_stages` stages, followed by decimation by `dsr`                                                | `ceil(N / dsr)`                 |
| `DownSamplingTargets.Polyphase`  | Cascade of `log2(dsr)` decimation-by-2 stages (`dsr` must be a power of two)                                                             | `floor(N / dsr)`                |

Details:

- **Subsampling**
  - `forward()` returns the samples at positions `0, dsr, 2*dsr, ...`.
  - With `do_subsampling()` you can select a different start offset (`take_sample`) or use data augmentation (see below).
- **Simple**
  - Acts as a moving-average filter combined with decimation, so it reduces noise and aliasing compared to pure subsampling.
- **CIC**
  - The input samples are rounded to integers before filtering (integer arithmetic as in the hardware implementation).
  - The output is normalized by the filter gain `dsr ** num_stages`.
- **Polyphase**
  - Each stage halves the sampling rate. The stage kernel is `[1, 2, 1]` (second order, used by `forward()`) or `[1, 1]` (first order).
  - The output is **not normalized**. The amplitude is scaled by the DC gain of the filter (4 per stage for second order, 2 per stage for first order).

### Subsampling with augmentation

```python
do_subsampling(data, augment=False, take_sample=0)
```

| Argument      | Description                                                                                                                              |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `data`        | Array with the high-rate signal. Subsampling is done along the last axis.                                                                |
| `augment`     | `True`: generates `dsr` sub-sampled versions (one per start offset) and concatenates them along the first axis. Shorter versions are zero-padded to equal length. |
| `take_sample` | Start offset (`0 … dsr-1`) of the returned version if `augment=False`                                                                   |

For `dsr == 1` the data is returned unchanged.

## Settings

Configuration is done with `DownSamplingSettings`:

| Attribute       | Type                 | Description                                                                          |
| --------------- | -------------------- | ------------------------------------------------------------------------------------ |
| `method`        | `DownSamplingTargets`| Downsampling method (`Subsampling`, `Simple`, `CIC`, `Polyphase`)                    |
| `sampling_rate` | `float`              | Sampling rate of the input data stream [Hz]                                          |
| `num_stages`    | `int`                | Number of CIC stages (also see note on the polyphase hardware design)                |
| `dsr`           | `int`                | Downsampling ratio (`SR_out = SR_in / dsr`)                                          |

A default configuration (`Simple`, 1000 Hz, 5 stages, `dsr=10`) is available as `DefaultDownSamplingSettings`.

## Usage

```python
import numpy as np

from elasticai.preprocessor.downsampling import (
    DownSampling,
    DownSamplingSettings,
    DownSamplingTargets,
)

settings = DownSamplingSettings(
    method=DownSamplingTargets.Simple,
    sampling_rate=1000.0,
    num_stages=1,
    dsr=10,
)

dut = DownSampling(settings=settings)
x = np.random.randn(1000)
y = dut.forward(x)                 # 100 samples
print(dut.sampling_rate_out)       # 100.0
```
An example for applying input data (top) and returning the downsampled data (bottom):
<p>
<img src="elasticai/preprocessor/downsampling/docu/example.svg" width="600">
</p>

Notes:

- Called with a `DataTransient` (`__call__`), the module returns a new `DataTransient` with the downsampled data and the sampling rate `sampling_rate / dsr`. The value of `sampling_rate` in the settings is used, not the rate of the incoming data, so it must match the input data.
- `method` can also be given as string (`"subsampling"`, `"simple"`, `"cic"`, `"polyphase"`).

### Hardware design generation

`create_design(id, target, path2save, total_bitwidth, frac_bitwidth=0, is_signed=True)` exports the module as hardware block.

| `target` (case-insensitive) | `Subsampling` | `Simple` | `CIC` | `Polyphase` |
| --------------------------- | :-----------: | :------: | :---: | :---------: |
| `"mcu"`, `"pc"` (C code)    | yes           | yes      | yes   | yes         |
| `"fpga"` (Verilog)          | yes           | yes      | yes   | yes         |
| `"asic"` (Verilog)          | no            | no       | no    | yes         |

Unsupported combinations raise a `NotImplementedError`.

Requirements and behaviour:

- `dsr >= 1` and `num_stages >= 1` are required.
- `total_bitwidth` must be between 2 and 32. It is the bit width of the data stream at the block.
- `is_signed` is only used for the C designs. `frac_bitwidth` is not used.
- `CIC`: `dsr` and `num_stages` are passed as `DEC_RATE` and `N_DEC`.
- `Polyphase`: for the C design the first-order variant is used if `num_stages` is odd, otherwise the second-order variant. In the Verilog designs `dsr` is passed as `POLY_ORDER`.