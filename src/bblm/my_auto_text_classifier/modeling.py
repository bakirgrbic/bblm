"""Implements my own generalizable AutoModel for text classification."""

import torch
from transformers import AutoConfig, AutoModel


class MyAutoModelTextClassifier(torch.nn.Module):
    def __init__(
        self, model_name: str, num_out: int, revision: str | None = None
    ) -> None:
        """
        Parameters
        ----------
        model_name
            relative file path of pretrained model or name from huggingface
        num_out
            number of classes to classify
        revision
            the specific commit of a model to use from huggingface.
        """
        super().__init__()
        self.transformer_layer = AutoModel.from_pretrained(
            model_name, revision=revision
        )
        # config needed to get appropriate hidden size for any model
        config = AutoConfig.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(config.hidden_size, num_out)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Runs a piece of tokenized data through the model.

        Parameters
        ----------
        input_ids
            input_ids from a transformer tokenizer
        attention_mask
            attention mask for input_ids from transformer tokenizer

        Returns
        -------
        the softmax distribution for all classes
        """
        output_transformer = self.transformer_layer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = output_transformer.last_hidden_state
        cls_pooler = last_hidden_state[:, 0]
        output = self.classifier(cls_pooler)

        return output
