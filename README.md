# bblm
A project to better undestand Language Models (LMs), pretraining, finetuning,
and modifying LMs. Inspired by the [BabyLM Challenge](https://babylm.github.io/index.html).

NOTE:
* Use huggingface models you trust. When possible specify the exact revision number for the model you are using.
This project is still a work in progress.
- electra-tiny-elc does not work with WoS task
- other huggingface models such as google-bert/bert-base-uncased do not work with the WoS task

## How to Run
Tested on arm64 MacOS and AWS Sagemaker environments.

Install the [conda](https://anaconda.org/) command line tool and use the following commands to simply run finetuning and pretraining scripts:
```shell
conda env create -f environment.yml
conda activate bblm
pip install .
```

### Install Data
Install the BabyLM 2024 training data and the Web of Science (WoS) using the following:
```shell
./src/bblm/cli/download_data.py
```
This is necessary for both running integration tests and using the training scripts.


### Running Tests
To run tests and other development tools, install them with the following command:
```shell
pip install -e ".[dev]"
```

To run unit tests and integration tests use the following:
```shell
pytest -m "not benchmark"
```
To only run slower performance tests use:
```shell
pytest -m "benchmark"
```

To run bandit locally on all files use the following command:
```shell
pre-commit run --hook-stage manual bandit --all-files
```

Proceed to [Pretraining](#pretraining) or [WoS Text Classification](#wos-text-classification) to begin training and evaluating models!


## Training
Make sure to install data as described in [Install Data](#install-data)

## Pretraining
To pretrain a local model or one from huggingface use the following command:
```shell
./training/pretrain.py [-h] [-m MODEL_NAME] [-r REVISION] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE] [-d DEVICE]
```

## WoS Text Classification
To finetune a local model or one from huggingface use the following command:
```shell
./training/finetune.py [-h] [-m MODEL_NAME] [-r REVISION] [-ml MAX_LEN] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE] [-d DEVICE]
```

## Acknowledgments
Code from [bblm/models/electra_tiny_elc.py](./bblm/models/electra_tiny_elc.py) is a derivative work of the following:
- the [transformers ELECTRA model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/electra/modeling_electra.py)
- the configuration of [bsu-slim/electra-tiny](https://huggingface.co/bsu-slim/electra-tiny)
- the [ELC-BERT zero initialization model](https://github.com/ltgoslo/elc-bert/blob/main/models/model_elc_bert_zero.py)

### Data
#### train_10M
Data used to pretrain models from the
[BabyLM 2024 data repository](https://osf.io/5mk3x).

#### wos
Web of Science text classification data used to fine-tune
and evaluate models. Specifically, the
[Web of Science Dataset WOS-46985](https://data.mendeley.com/datasets/9rw3vkcfy4/6)
data. Also found on [huggingface](https://huggingface.co/datasets/bakirgrbic/web-of-science).
