# Spectral-Symphonies

## Table of Contents

<!-- - [Spectral-Symphonies](#spectral-symphonies) -->
<!--   - [Table of Contents](#table-of-contents) -->
  - [Project Overview](#project-overview)
  - [Installation](#installation)
  - [Implementation](#implementation)
  - [Features](#features)
  - [Credits](#credits)

  <!-- - [Contributing](#contributing) -->
  <!-- - [License](#license) -->

## Project Overview

In this paper, we propose a Mandarin speaker recognition system that utilises spectral features, including Mel-Frequency Cepstral Coefficients (MFCC), Delta-Mel-Frequency Cepstral Coefficients (DMFCC) and Filter Bank Coefficients (FBC). We employ a conditional Generative Adversarial Network (CGAN), conditioned on known speaker labels, for the speaker classification. We evaluated our proposed system using the THCHS-30 dataset, a labelled Mandarin audio dataset. The results of our evaluation are promising, demonstrating the system’s ability to identify and distinguish speakers both within and outside the domain.

## Installation

1.  Create a Conda Virtual Environment (named *SS*) & activate the Virtual Environment:
```
conda create -n SS python=3.9.10
conda activate SS
```
2.  Install required packages:
```
pip install numpy scipy tensorflow==2.10.1 keras==2.10.0 tqdm sklearn matplotlib pandas
```

## Implementation

We provide the pre-processing code for users to use directly:
<a href="https://colab.research.google.com/drive/1qavUj0obBT_kOOzNZ6WU4uAug3fqTYEK?usp=sharing">
  <img src="Figs/colab.png" alt="Open In Colab" style="width:30px;height:30px;">
</a>

*Click on the Colab icon to navigate to the Google Colab interface!*
<br><br>

In our project, we support two different running modes, including **train** and **test**. Some example command-line instructions are provided below.

### Train from scratch
```
python main.py --mode train --load-npz-dir .
```

### Train with loaded model
For example, if we load model from `saved_models` at epoch 18000:
```
python main.py --mode train --load-npz-dir . --load-ckpt saved_models --ckpt_epoch 18000
```

### Train with incomplete features
For example, if we use DMCC + MFCC:
```
python main.py --mode train --load-npz-dir . --selected-features DMFCC MFCC
```

### Test with loaded model
```
python main.py --mode test --load-npz-dir . -load-ckpt saved_models
```

In addition to those, we also have the following setups.

| Argument                | Description                                                          | Default Value  |
|-------------------------|----------------------------------------------------------------------|----------------|
| `--mode`                | Mode: train or test                                                  |                |
| `--load-npz-dir`        | Path to load `train.npz` and `test.npz`                              |                |
| `--load-ckpt`           | Directory that stores model checkpoint to load                       | ""             |
| `--ckpt_epoch`          | The epoch number for resuming training and Testing                   | None           |
| `--sample-size`         | Sampling size                                                        | 200            |
| `--epoch-num`           | Number of epochs                                                     | 20000          |
| `--class-num`           | Number of classes                                                    | 25             |
| `--batch-size`          | Batch size                                                           | 1024           |
| `--L-reduced-dim`       | Reduced dimension for L (None for original dimension)                | 78             |
| `--F-reduced-dim`       | Reduced dimension for F (None for original dimension)                | None           |
| `--selected-features`   | Names of selected features (choices: MFCC, DMFCC, FBC)               | MFCC DMFCC FBC |



## Features

In our paper, the experiments in Section 4.2 were conducted by setting different 
`--L-reduced-dim` {26,52,78,104,208,312},  `--F-reduced-dim` {3,6,9,12,15,18} to test different dimensionality reduction experiments; experiments in Section 4.3 were conducted by changing different combination for the `--selected-features` {MFCC, DMFCC, FBC, MFCC DMFCC, MFCC FBC, DMFCC FBC, MFCC DMFCC, FBC} arguments.

<!-- ## Contributing -->


<!-- ## License -->

<!-- [Specify the license under which your project is distributed. Include any additional terms or permissions.]

[Optional: Add any acknowledgements, credits, or references to external resources.] -->

## Credits
Credits to https://github.com/eriklindernoren/Keras-GAN for which our CGAN code is built on.
