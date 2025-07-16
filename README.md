# bblm
A project to better undestand Language Models (LMs), pretraining, finetuning, 
and modifying LMs. Inspired by the [BabyLM Challenge](https://babylm.github.io/index.html).

NOTE:
This project is still a work in progress.
- electra-tiny-elc does not work with WOS finetuning task

## How to Run
Tested on arm64 MacOS and AWS Sagemaker environments. 
0. Set up environment
    - Make sure you have some version of conda and use the following command:
    ```shell
    make env
    ```
    - If running on AWS Sagemaker, install `transformers` and `torch` libraries using pip. 
1. Download necessary data as described in [data/README.md](./data/README.md).
2. Optionally run tests
    - Use the following to run unit tests and to check data is downloaded and in the right spot.
    ```shell
    make fast-tests
    ```
    - Use the following to run slower tests that are useful for checking pipelines are not throwing any errors
    ```shell
    make slow-tests
    ```
3. Use the commands below to begin pretraining or finetuning models!

## Pretraining

To pretrain a local model or one from huggingface use the following command:
```shell
python3 -m src.pretrain.py [-h] [-m MODEL_NAME] [-t TOKENIZER_NAME] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE]
```

## Finetuning
### Web of Science (WOS) Text Classification

To finetune a local model or one from huggingface use the following command:
```shell
python3 -m src.finetune.py [-h] [-m MODEL_NAME] [-t TOKENIZER_NAME] [-ml MAX_LEN] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE]
```

## Acknowledgments
Code from [src/models/electra_elc.py](./models/electra_elc.py) is a derivative work
of [transformers ELECTRA model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/electra/modeling_electra.py) 
and [ELC-BERT zero initialization model](https://github.com/ltgoslo/elc-bert/blob/main/models/model_elc_bert_zero.py).
