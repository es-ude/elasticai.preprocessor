# elasticAI.preprocessor - Python Extension for Enabling Sensor Data Pre-Processing on Software and Hardware
This Python framework allows to deploy preprocessing methods on transient data. This can be done on pre-recorded data (offline) for evaluating an user-specific end-to-end signal processor.
Afterwards, the used methods with its configuration can be deployed on hardware (MCU / FPGA) for real-time signal processing. We deliver the hardware-software co-design.

In general, this framework aims to accelerate the deployment of signal processor pipelines in next-gen hardware development. 
For building the whole digital AI hardware, we recommend to use the following framework: 
- [denspp.offline](https://github.com/es-ude/denspp.offline): Finding the best preprocessing methods with offline data
- [elasticAI.explorer](https://github.com/es-ude/elastic-ai.explorer): Hardware-aware NAS
- [elasticAI.creator](https://github.com/es-ude/elastic-ai.creator): Building embedded deep learning models
- [elasticAI.hardware](https://github.com/es-ude/elastic-ai.hardware): Hardware for deploying the embedded models

# Table of Content
1. [Installation guide](#installation-guide)
2. [Citation](#citation--documentation)

## Installation guide
For using this framework, the following software tools are necessary / recommended.
- `uv` package manager ([Link](https://docs.astral.sh/uv/), [Using](https://www.saaspegasus.com/guides/uv-deep-dive/))
- Git ([Link](https://git-scm.com/downloads))

It is recommended that each new feature will be edited in a new branch. If the integration is done and all tests are runned successful, please create a pull request for merging it back into the main branch. Further information about using this software framework are described in the paper at the end of the readme file.

If you create a new repo and you want to use the functionalities of this Python tool. Then please initialise the project.toml using `uv` and write the following code into the terminal.
```
uv add "git+https://github.com/es-ude/elasticai.preprocessor.git"
```
Afterwards you can create the virtual environment (venv) and installing all packages using this line.
````
uv sync (--refresh --upgrade)
````
## Citation / Documentation
If you want to understand what is the idea behind elasticAI ecosystem, please have a look on the corresponding [paper](https://doi.org/10.1515/cdbme-2023-1118) regarding end-to-end signal processing.
