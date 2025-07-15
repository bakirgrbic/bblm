# bblm

## Description
A project to better undestand Language Models (LMs), pre-training, finetuning, 
and modifying LMs. Inspired by the [BabyLM Challenge](https://babylm.github.io/index.html).

NOTE:
This project is still a work in progress.
- electra-tiny-elc does not work with WOS finetuning task

## How to Run
Tested on arm64 MacOS and AWS Sagemaker environments. 
When running on AWS Sagemaker make sure to install the `transformers` and 
`torch` libraries. To use any pretrained models from this repository do the following:
1. Install [git-lfs](https://git-lfs.com/)
2. Run the following commands to pull the four models
```shell
git lfs fetch --all
```
```shell
git lfs checkout
```

### Data
Make sure to download necessary data as described in [data/README.md](./data/README.md).

### Pretraining

To pretrain a local model or one from huggingface use the following command:
```shell
python3 -m src.pretrain.py [-h] [-m MODEL_NAME] [-t TOKENIZER_NAME] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE]
```

### Finetuning
#### Web of Science (WOS) Text Classification

To finetune a local model or one from huggingface use the following command:
```shell
python3 -m src.finetune.py [-h] [-m MODEL_NAME] [-t TOKENIZER_NAME] [-ml MAX_LEN] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE]
```

## Acknowledgments
Code from [src/models/electra_elc.py](./models/electra_elc.py) is a derivative work
of [transformers ELECTRA model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/electra/modeling_electra.py) 
and [ELC-BERT zero initialization model](https://github.com/ltgoslo/elc-bert/blob/main/models/model_elc_bert_zero.py).
