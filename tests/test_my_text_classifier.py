import pytest
from transformers import AutoTokenizer

from bblm.my_text_classifier.configuration import MyConfig
from bblm.my_text_classifier.modeling import MyTextClassifier


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
def test_model_forward_pass_raise_no_error(model_name, sample_data):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = MyConfig(model_name, num_classes=2)
    model = MyTextClassifier(config)

    inputs = tokenizer(
        sample_data,
        return_tensors="pt",
    )
    model(inputs["input_ids"], inputs["attention_mask"])


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
def test_config_save_no_error(model_name, tmp_path):
    config = MyConfig(model_name=model_name, num_classes=2)

    config.save_pretrained(tmp_path)

    assert (tmp_path / "config.json").exists()


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
def test_config_load_no_error(model_name, tmp_path):
    config = MyConfig(model_name=model_name, num_classes=2)

    config.save_pretrained(tmp_path)

    MyConfig.from_pretrained(tmp_path)


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
def test_model_save_no_error(model_name, tmp_path):
    config = MyConfig(model_name, num_classes=2)
    model = MyTextClassifier(config)

    model.save_pretrained(tmp_path)

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()


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
def test_model_load_no_error(model_name, tmp_path):
    config = MyConfig(model_name, num_classes=2)
    model = MyTextClassifier(config)

    model.save_pretrained(tmp_path)

    MyTextClassifier.from_pretrained(tmp_path)
