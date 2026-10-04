# Overview of supported layers

The following layers can be used to apply for time series analysis.

The error classes are defined as below:
- 0: Hardware matches the quantized forward path totally 
- 1: Hardware matches the quantized forward path with given tolerance range
- n.A.: not available

## Methods for Pre-Processing Data Streams
The named layers are used to enable a hardware-aware time series analysis with deploying on different hardware types.

| Type                                | Subpackage name    | Function                                                                    | Error class                | Supported hardware types         |
|:------------------------------------|:-------------------|:----------------------------------------------------------------------------|:---------------------------|:---------------------------------|
| **Rescaler**                        | *adc*              | Rescaling the input data stream (from ADC to model)                         | n.A.                       | not supported yet                |
| **Downsampling**                    | *downsampling*     | Adapting the bit resolution and sampling rate of the input data stream      | 0                          | FPGA, MCU, Workstation           | 
| **Referencing**                     | *referencing*      | Artefact suppress technique like CAR for multi-channel processing           | n.A.                       | not supported                    |
| **Filtering**                       | *filter*           | Applying data filtering on data stream (FIR, IIR)                           | FIR: 0, IIR: 1 (+/- 1 LSB) | FPGA, MCU, Workstation           |
| **Event-Detecion**                  | *eventdetection*   | Apply a comparator on a data stream (normal, hysteresis)                    | 0                          | FPGA, MCU, Workstation           |
| **FrameAligner**                    | *eventdetection*   | Aligning all detected frames to specified technique (min, max, ...)         | n.A.                       | not supported                    |
| **Preprocessor**                    | *eventdetection*   | Preprocessor Method (normal, abs, NEO) to comparator input to increase SNR  | 0                          | MCU, Workstation                 |
| **Thresholding**                    | *thresholding*     | Extract threshold values from input data stream for event-detection         | 1 (+/- 1 LSB)              | FPGA, MCU, Workstation           |
| **Normalization**                   | *normalization*    | Apply a data normalization (min-max, ...) on data segments to apply to DNNs | 1 (+/- 1 LSB)              | MCU, Workstation (all partially) |
| **Transformation**                  | *transformation*   | Apply a time-to-frequence transformation (FFT, Wavelet, ...)                | n.A.                       | n.A.                             |
| **Windower**                        | *windower*         | Extracting windows from data stream (sliding, sequence, event)              | 0                          | FPGA                             |
| **Spike Detection Algorithm (SDA)** | *sda*              | Extracting event-based and aligned windows                                  | n.A.                       | not supported yet                |

## Methods for Data Adaption / Augmentation
The following layers are used to generate data signals in order to fed-in in the pre-processing pipeline or other use-cases.

| Type                    | Subpackage name | Function                                                                                     | Error class     | Supported hardware types |
|:------------------------|:----------------|:---------------------------------------------------------------------------------------------|:----------------|:-------------------------|
| **ConversionResampler** | *adc*           | Requantization the input data stream (data can be replayed via Testbench or MCU in hardware) | 0               | FPGA, MCU, Workstation   |
| **Augmentation**        | *downsampling*  | Augmentating the data based on subsampling downsampling                                      | n.A.            | will not be supported    |
| **Waveform Generator**  | *waveform*      | Generating a LUT-based waveform to run                                                       | 0               | FPGA, MCU, Workstation   |    
