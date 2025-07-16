# bblm
A project to better undestand Language Models (LMs), pretraining, finetuning, 
and modifying LMs. Inspired by the [BabyLM Challenge](https://babylm.github.io/index.html).

NOTE:
This project is still a work in progress.
- electra-tiny-elc does not work with WOS finetuning task

## How to Run
Tested on arm64 MacOS and AWS Sagemaker environments.

Install the [conda](https://anaconda.org/) command line tool and use the following command:
```shell
make env
```
If running on AWS Sagemaker, install the following libraries with pip:
- `transformers`
- `torch`
- `pytest` 

Next, download necessary data as described in [data/README.md](./data/README.md). To run
faster unit tests and to check necessary data is downloaded in the right folder, use the following:
```shell
make fast-tests
```
To only run slower tests that are useful for checking if pipelines are throwing any errors use:
```shell
make slow-tests
```
To run all unit tests use the following command:
```shell
make all-tests
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
Code from [src/models/electra_elc.py](./models/electra_elc.py) is a derivative work
of [transformers ELECTRA model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/electra/modeling_electra.py) 
and [ELC-BERT zero initialization model](https://github.com/ltgoslo/elc-bert/blob/main/models/model_elc_bert_zero.py).
