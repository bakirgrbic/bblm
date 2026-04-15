"""Implements config for MyTextClassifier"""

from transformers import PreTrainedConfig


class MyConfig(PreTrainedConfig):
    def __init__(
        self,
        model_name: str = "bakirgrbic/electra-tiny",
        num_classes: int = 7,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        super().__init__(**kwargs)
