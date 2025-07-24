# bblm
A project to better undestand Language Models (LMs), pretraining, finetuning, 
and modifying LMs. Inspired by the [BabyLM Challenge](https://babylm.github.io/index.html).

NOTE:
This project is still a work in progress.
- electra-tiny-elc does not work with WOS task
- other huggingface models such as google-bert/bert-base-uncased do not work with the WOS task

## How to Run
Tested on arm64 MacOS and AWS Sagemaker environments.

Install the [conda](https://anaconda.org/) command line tool and use the following command:
```shell
conda env create -f environment.yml
```
If running on AWS Sagemaker, install the following libraries with pip:
- `transformers`
- `torch`
- `pytest` 

Next, download necessary data as described in [data/README.md](./data/README.md). To run
unit tests and smoke tests use the following:
```shell
python3 -m pytest -m "not benchmark"
```
To only run slower performance tests use:
```shell
python3 -m pytest -m "benchmark"
```

Proceed to [pretraining](#pretraining) or [finetuning](#finetuning) to begin training and evaluating models!


## Pretraining

To pretrain a local model or one from huggingface use the following command:
```shell
python3 -m bblm.pretrain.py [-h] [-m MODEL_NAME] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE]
```

## Finetuning
### Web of Science (WOS) Text Classification

To finetune a local model or one from huggingface use the following command:
```shell
python3 -m bblm.finetune.py [-h] [-m MODEL_NAME] [-ml MAX_LEN] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE]
```

## Acknowledgments
Code from [bblm/models/electra_tiny_elc.py](./bblm/models/electra_tiny_elc.py) is a derivative work of the following:
- the [transformers ELECTRA model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/electra/modeling_electra.py)
- the configuration of [bsu-slim/electra-tiny](https://huggingface.co/bsu-slim/electra-tiny)
- the [ELC-BERT zero initialization model](https://github.com/ltgoslo/elc-bert/blob/main/models/model_elc_bert_zero.py)
