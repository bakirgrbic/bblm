import pytest
from transformers import AutoTokenizer

from bblm.my_auto_text_classifer import MyAutoModelTextClassifier


@pytest.fixture
def sample_data():
    return "Doesn't matter so long as it can be passed forward!"


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param(
            "bakirgrbic/electra-tiny",
            id="electra",
        ),
        pytest.param(
            "google-bert/bert-base-uncased",
            id="bert",
        ),
    ],
)
def test_forward_pass_raise_no_error(model_name, sample_data):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = MyAutoModelTextClassifier(model_name, num_out=2)

    inputs = tokenizer(
        sample_data,
        return_tensors="pt",
    )
    model(inputs["input_ids"], inputs["attention_mask"])
