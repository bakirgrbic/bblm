"""Implements my own model for text classification."""

import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel

from bblm.my_text_classifier.configuration import MyConfig


class MyTextClassifier(PreTrainedModel):
    def __init__(self, config: MyConfig) -> None:
        super().__init__(config)

        self.transformer_layer = AutoModel.from_pretrained(config.model_name)

        # redeundant config needed to get appropriate hidden size for any model
        config_for_hidden = AutoConfig.from_pretrained(config.model_name)
        self.classifier = torch.nn.Linear(
            config_for_hidden.hidden_size, config.num_classes
        )

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
