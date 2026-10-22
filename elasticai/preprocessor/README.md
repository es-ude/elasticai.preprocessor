# Overview of supported layers

The following layers can be used to apply for time series analysis.

## Methods for Pre-Processing Data Streams
The named layers are used to enable a hardware-aware time series analysis with deploying on different hardware types.
The layers are sorted for the position to use in the processing stage. We recommend to follow the instruction orders.

| Type                                                 | Subpackage name   | Function                                                                                      
|:-----------------------------------------------------|:------------------|:----------------------------------------------------------------------------------------------|
| [**ConversionResampler**](adc/README.md)            | *adc*             | Requantization the input data stream (data can be replayed via Testbench or MCU in hardware)  |
| [**Rescaler**](adc/README.md)                        | *adc*             | Rescaling the input data stream (from ADC to model)                                           |
| [**Downsampling**](downsampling/README.md)           | *downsampling*    | Adapting the bit resolution and sampling rate of the input data stream                        | 
| [**Referencing**](referencing/README.md)             | *referencing*     | Artefact suppress technique like CAR for multi-channel processing                             |
| [**Filtering**](filter/README.md)                    | *filter*          | Applying data filtering on data stream (FIR, IIR)                                             | 
| [**Event-Detecion**](eventdetection/README.md)       | *eventdetection*  | Apply a comparator on a data stream (normal, hysteresis)                                      | 
| [**FrameAligner**](eventdetection/README.md)         | *eventdetection*  | Aligning all detected frames to specified technique (min, max, ...)                           | 
| [**Preprocessor**](eventdetection/README.md)         | *eventdetection*  | Preprocessor Method (normal, abs, NEO) to comparator input to increase SNR                    | 
| [**Thresholding**](thresholding/README.md)           | *thresholding*    | Extract threshold values from input data stream for event-detection                           | 
| [**Normalization**](normalization/README.md)         | *normalization*   | Apply a data normalization (min-max, ...) on data segments to apply to DNNs                   | 
| [**Transformation**](transformation/README.md)       | *transformation*  | Apply a time-to-frequence transformation (FFT, Wavelet, ...)                                  | 
| [**Windower**](windower/README.md)                   | *windower*        | Extracting windows from data stream (sliding, sequence, event)                                | 
| [**Spike Detection Algorithm (SDA)**](sda/README.md) | *sda*             | Extracting event-based and aligned windows                                                    | 

## Methods for Data Adaption / Augmentation
The following layers are used to generate data signals in order to fed-in in the pre-processing pipeline or other use-cases.

| Type                                          | Subpackage name | Function                                                                                     |
|:----------------------------------------------|:----------------|:---------------------------------------------------------------------------------------------|
| **Augmentation**                              | *downsampling*  | Augmentating the data based on subsampling downsampling                                      |
| [**Waveform Generator**](wavform/README.md)   | *waveform*      | Generating a LUT-based waveform to run                                                       | 
