# Wavehax

This repository provides the official PyTorch implementation of [Wavehax](https://chomeyama.github.io/wavehax-demo/), an alias-free neural vocoder that combines 2D convolutions with harmonic priors for high-fidelity and robust complex spectrogram estimation.


## Environment Setup

To set up the environment, run:
```bash
$ cd wavehax
$ pip install -e .
```
This will install the necessary dependencies in editable mode.


## Directory structure

- **egs**:
This directory contains project-specific examples and configurations.
- **egs/jvs**:
An example project using the [Japanese Versatile Speech (JVS) Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus), with speaker- and style-wise fundamental frequency (F0) ranges available at [JVS Corpus F0 Range](https://github.com/chomeyama/JVSCorpusF0Range).
- **wavehax**:
The main source code for Wavehax.


## Run

This repository uses [Hydra](https://hydra.cc/docs/intro/) for managing hyperparameters.
Hydra provides an easy way to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

### Dataset preparation

Prepare your dataset by creating `.scp` files that define the path to each audio file (e.g., `egs/jvs/data/scp/train_no_dev.scp`).
During the preprocessing step, list files for the extracted features will be automatically generated (e.g., `egs/jvs/data/list/train_no_dev.list`).
Ensure that separate `.scp` and `.list` files are available for training, validation, and evaluation datasets.


### Preprocessing

To extract acoustic features and prepare statistics:
```bash
# Move to the project directory.
$ cd egs/jvs

# Extract acoustic features like F0 and mel-spectrogram. To customize hyperparameters, edit wavehax/bin/config/extract_features.yaml, or override them from the command line.
$ wavehax-extract-features audio=data/scp/jvs_all.scp

# Compute statistics of the training data. You can adjust hyperparameters in wavehax/bin/config/compute_statistics.yaml.
$ wavehax-compute-statistics feats=data/scp/train_no_dev.list stats=data/stats/train_no_dev.joblib
```

### Training

To train the vocoder model:
```bash
# Start training. You can adjust hyperparameters in wavehax/bin/config/decode.yaml. In the paper, the model was trained for 1000K steps to match other models, but Wavehax achieves similar performance with fewer training steps.
$ wavehax-train generator=wavehax discriminator=univnet train=wavehax train.train_max_steps=500000 data=jvs out_dir=exp/wavehax
```

### Inference

To generate speech waveforms using the trained model:
```bash
# Perform inference using the trained model. You can adjust hyperparameters in wavehax/bin/config/decode.yaml.
$ wavehax-decode generator=wavehax data=jvs out_dir=exp/wavehax ckpt_steps=500000
```

### Monitoring training progress

You can monitor the training process using [TensorBoard](https://www.tensorflow.org/tensorboard):
```bash
$ tensorboard --logdir exp
```


### Pretrained models

We plan to release models trained on several datasets.
