# DataNormalization

Preprocessing module that **normalizes the amplitude of the input data** to improve the performance of the following classification or processing stages. Six normalization methods are available. They work on `numpy` arrays and `torch` tensors. For MCU/PC, a hardware design (C code) can be generated for the methods `minmax` and `zscore`.

## How it works

The normalization is applied **along the last axis** (the sample axis) and **separately for each signal**. The module is stateless: all parameters (peak value, mean, standard deviation, ...) are determined from the data of each call. There are no fixed or trained parameters.

| `method`                               | Calculation                                                    | Output range / property            |
| -------------------------------------- | -------------------------------------------------------------- | ---------------------------------- |
| `DataNormalizationTargets.Minmax`      | `x / peak`, with `peak = max(abs(x))`                          | `[-1, 1]`                          |
| `DataNormalizationTargets.Zeroone`     | `0.5 + x / (2 * peak)`, with `peak = max(abs(x))`              | `[0, 1]`                           |
| `DataNormalizationTargets.Norm`        | `x / ‖x‖₂` (Euclidean norm)                                    | unit norm                          |
| `DataNormalizationTargets.Zscore`      | `(x - mean) / std` (population standard deviation)            | zero mean, unit standard deviation |
| `DataNormalizationTargets.MedinanMAD`  | `(x - median) / MAD`, with `MAD = median(abs(x - median))`     | robust against outliers            |
| `DataNormalizationTargets.MeanMAD`     | `(x - mean) / MAD`, with `MAD = mean(abs(x - mean))`           | zero mean                          |

Notes:

- `Minmax` scales with the peak value only (no offset removal). It is **not** the `(x - min) / (max - min)` scaling.
- `MedinanMAD` is the (misspelled) name of the enum member for the string `"medianmad"`.
- There is no protection against division by zero. A constant signal (standard deviation, MAD or norm equal to zero) results in `inf` or `NaN`.

### Peak mode

`peak_mode` selects how the peak value is determined:

| `peak_mode`                          | Peak value        |
| ------------------------------------ | ----------------- |
| `DataNormalizationPeakMode.Max` (0)  | `max(x)`          |
| `DataNormalizationPeakMode.Min` (1)  | `abs(min(x))`     |
| `DataNormalizationPeakMode.AbsMax` (2) | `max(abs(x))`   |

The methods `Minmax` and `Zeroone` currently always use `AbsMax` and set `peak_mode` in the settings to `AbsMax` when `forward()` is called. All other methods do not use the peak value.

### Input shapes

| Input shape                  | Result                                            |
| ---------------------------- | ------------------------------------------------- |
| `(num_samples,)`             | One signal, normalized as a whole                 |
| `(num_signals, num_samples)` | Each row is normalized independently              |
| `(..., num_samples)`         | Each signal along the last axis is normalized independently |

The output has the same shape and type (`numpy` / `torch`) as the input.

## Settings

Configuration is done with `DataNormalizationSettings`:

| Attribute   | Type                                  | Description                                                           |
| ----------- | ------------------------------------- | --------------------------------------------------------------------- |
| `method`    | `DataNormalizationTargets`            | Normalization method (`Norm`, `MedinanMAD`, `MeanMAD`, `Minmax`, `Zscore`, `Zeroone`) |
| `peak_mode` | `int` or `DataNormalizationPeakMode`  | Mode for taking the peak value (0: max, 1: min, 2: abs-max)           |

`method` can also be given as lowercase string (`"norm"`, `"medianmad"`, `"meanmad"`, `"minmax"`, `"zscore"`, `"zeroone"`).

A default configuration (`Minmax`, `AbsMax`) is available as `DataNormalizationSettingsDefault`.

## Usage

```python
import numpy as np

from elasticai.preprocessor.normalization import (
    DataNormalization,
    DataNormalizationPeakMode,
    DataNormalizationSettings,
    DataNormalizationTargets,
)

settings = DataNormalizationSettings(
    method=DataNormalizationTargets.Zscore,
    peak_mode=DataNormalizationPeakMode.AbsMax,
)

dut = DataNormalization(settings=settings)

x = np.random.randn(10, 500)       # 10 signals with 500 samples each
y = dut.forward(x)                 # each row has zero mean and unit standard deviation
```
An example for applying input data (top) and returning the filtered data (bottom):
<p>
<img src="elasticai/preprocessor/normalization/docu/example.svg" width="600">
</p>
With `torch` tensors:

```python
import torch

y = dut.forward(torch.randn(10, 500))
```

Notes:

- Called with a `DataTransient` (`__call__`), the module returns a `DataTransient` with the normalized data and the **unchanged sampling rate**.

### Hardware design generation

`create_design(id, target, path2save, total_bitwidth, frac_bitwidth=0, is_signed=True)` exports the module as hardware block.

| `target` (case-insensitive) | Status                                                      |
| --------------------------- | ----------------------------------------------------------- |
| `"mcu"`, `"pc"`             | C code for `Minmax` (only with `peak_mode=AbsMax`) and `Zscore` |
| `"fpga"`                    | not implemented yet (`NotImplementedError`)                 |

Requirements and behaviour:

- Other normalization methods than `Minmax` and `Zscore` raise a `NotImplementedError` for the C design.
- `total_bitwidth` and `is_signed` define the data type of the generated block. `frac_bitwidth` is not used.
- Unsupported targets raise a `ValueError`.