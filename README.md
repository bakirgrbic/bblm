# bblm
A project to better undestand Language Models (LMs), pretraining, finetuning,
and modifying LMs. Inspired by the [BabyLM Challenge](https://babylm.github.io/index.html).

NOTE:
* Use huggingface models you trust. When possible specify the exact revision number for the model you are using.
This project is still a work in progress.
- electra-tiny-elc does not work with WOS task
- other huggingface models such as google-bert/bert-base-uncased do not work with the WOS task

## How to Run
Tested on arm64 MacOS and AWS Sagemaker environments.

Install the [conda](https://anaconda.org/) command line tool and use the following commands to simply run finetuning and pretraining scripts:
```shell
conda env create -f environment.yml
conda activate bblm
pip install .
```

### Running Tests
To run tests and other development tools, install them with the following command:
```shell
pip install -e ".[dev]"
```

Next, download necessary data as described in [data/README.md](./data/README.md). To run
unit tests and smoke tests use the following:
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

Proceed to [pretraining](#pretraining) or [finetuning](#finetuning) to begin training and evaluating models!


## Pretraining

To pretrain a local model or one from huggingface use the following command:
```shell
./training/pretrain.py [-h] [-m MODEL_NAME] [-r REVISION] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE] [-d DEVICE]
```

## Finetuning
### Web of Science (WOS) Text Classification

To finetune a local model or one from huggingface use the following command:
```shell
./training/finetune.py [-h] [-m MODEL_NAME] [-r REVISION] [-ml MAX_LEN] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE] [-d DEVICE]
```

## Acknowledgments
Code from [bblm/models/electra_tiny_elc.py](./bblm/models/electra_tiny_elc.py) is a derivative work of the following:
- the [transformers ELECTRA model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/electra/modeling_electra.py)
- the configuration of [bsu-slim/electra-tiny](https://huggingface.co/bsu-slim/electra-tiny)
- the [ELC-BERT zero initialization model](https://github.com/ltgoslo/elc-bert/blob/main/models/model_elc_bert_zero.py)
