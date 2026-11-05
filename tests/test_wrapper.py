import base64
from datetime import datetime
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest
import torch

from vial_classifier import config
from vial_classifier.modeling.model import SimpleCNN
from vial_classifier.modeling.wrapper import SimpleLoggingWrapper


class FixedProbabilityModel(torch.nn.Module):
    """Predicts the same probability for every image."""

    def __init__(self, probability=0.8):
        super().__init__()
        self.logit = torch.logit(torch.tensor(probability))

    def forward(self, x):
        return self.logit.expand(x.shape[0], 1)


@pytest.fixture(autouse=True)
def run_in_tmp_dir(tmp_path, monkeypatch):
    """The wrapper's default log path is relative, so keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "logs" / "inference_log.json"


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "vial.jpg"
    Image.new("RGB", (32, 16), color=(120, 60, 30)).save(path, format="JPEG")
    return path


@pytest.fixture
def wrapper(log_path):
    return SimpleLoggingWrapper(model=FixedProbabilityModel(), log_path=log_path)


def _as_bytes(path):
    return path.read_bytes()


def _as_base64(path):
    return base64.b64encode(path.read_bytes()).decode()


def _read_log(path):
    return json.loads(path.read_text())


# --- input formats -----------------------------------------------------------


@pytest.mark.parametrize(
    "make_input",
    [
        pytest.param(_as_bytes, id="bytes"),
        pytest.param(_as_base64, id="base64"),
        pytest.param(str, id="str_path"),
        pytest.param(Path, id="path"),
        pytest.param(lambda p: {"instances": [_as_base64(p)]}, id="instances"),
        pytest.param(lambda p: {"data": [_as_base64(p)]}, id="data"),
        pytest.param(lambda p: {"image": _as_base64(p)}, id="other_key"),
        pytest.param(lambda p: [_as_base64(p)], id="list"),
        pytest.param(lambda p: (_as_base64(p),), id="tuple"),
        pytest.param(lambda p: np.array([_as_base64(p)]), id="numpy"),
    ],
)
def test_accepts_every_input_format(wrapper, image_path, make_input):
    result = wrapper.predict(None, make_input(image_path))

    assert result == {
        "predictions": [1],
        "confidences": [0.8],
        "raw_outputs": [0.8],
        "num_samples": 1,
    }


def test_nested_instances_are_flattened(wrapper, image_path):
    encoded = _as_base64(image_path)

    result = wrapper.predict(None, {"instances": [[encoded, encoded], encoded]})

    assert result["num_samples"] == 3
    assert result["predictions"] == [1, 1, 1]


@pytest.mark.parametrize("key", ["instances", "data"])
def test_contract_keys_win_over_other_keys(wrapper, image_path, key):
    result = wrapper.predict(None, {"request_id": "abc", key: [_as_base64(image_path)]})

    assert result["predictions"] == [1]


def test_dict_without_usable_input_raises(wrapper):
    with pytest.raises(ValueError, match="Could not extract"):
        wrapper.predict(None, {"number": 3})


def test_preprocess_returns_a_batch_of_one_resized_image(wrapper, image_path):
    assert wrapper.preprocess_image(image_path).shape == (1, 3, *config.IMAGE_SIZE)


# --- predictions and threshold -----------------------------------------------


def test_default_threshold_is_not_the_tuned_one(wrapper):
    # Notebook quirk: the logged wrapper keeps 0.5 and never receives the tuned threshold
    assert wrapper.threshold == 0.5


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [(0.51, 0.5, 1), (0.49, 0.5, 0), (0.51, 0.6, 0), (0.1, 0.025, 1)],
)
def test_prediction_follows_the_threshold(log_path, image_path, probability, threshold, expected):
    wrapper = SimpleLoggingWrapper(
        model=FixedProbabilityModel(probability), log_path=log_path, threshold=threshold
    )

    assert wrapper.predict(None, image_path)["predictions"] == [expected]


def test_confidence_is_rounded_in_the_response_but_not_in_the_log(log_path, image_path):
    wrapper = SimpleLoggingWrapper(model=FixedProbabilityModel(0.51444), log_path=log_path)

    result = wrapper.predict(None, image_path)

    assert result["confidences"] == [0.5144]
    assert result["raw_outputs"] == [0.5144]
    assert _read_log(log_path)[0]["confidence"] == pytest.approx(0.51444, abs=1e-6)


# --- bad input ---------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input",
    [
        pytest.param("not a path and not base64", id="bad_string"),
        pytest.param(b"not an image", id="bad_bytes"),
        pytest.param(Path("missing.jpg"), id="missing_path"),
        pytest.param(123, id="unsupported_type"),
    ],
)
def test_bad_input_gives_none_instead_of_crashing(wrapper, log_path, bad_input):
    result = wrapper.predict(None, [bad_input])

    assert result == {
        "predictions": [None],
        "confidences": [None],
        "raw_outputs": [None],
        "num_samples": 1,
    }
    assert not log_path.exists()


def test_bad_input_keeps_its_position_among_good_inputs(wrapper, log_path, image_path):
    encoded = _as_base64(image_path)

    result = wrapper.predict(None, [encoded, b"not an image", encoded])

    assert result["predictions"] == [1, None, 1]
    assert result["num_samples"] == 3
    assert len(_read_log(log_path)) == 2


# --- inference log -----------------------------------------------------------


def test_each_request_is_appended_to_the_log(wrapper, log_path, image_path):
    wrapper.predict(None, image_path)
    wrapper.predict(None, [image_path, image_path])

    assert len(_read_log(log_path)) == 3


def test_log_entry_has_what_drift_detection_reads(wrapper, log_path, image_path):
    wrapper.predict(None, image_path)

    entry = _read_log(log_path)[0]
    assert set(entry) == {"timestamp", "image_path", "prediction", "confidence", "raw_output"}
    assert entry["image_path"] == str(image_path)
    assert entry["prediction"] == 1
    # Drift detection compares against a timezone-aware time, so a naive one would crash
    assert datetime.fromisoformat(entry["timestamp"]).tzinfo is not None


def test_raw_bytes_are_logged_with_a_placeholder(wrapper, log_path, image_path):
    wrapper.predict(None, image_path.read_bytes())

    assert _read_log(log_path)[0]["image_path"] == "<bytes>"


@pytest.mark.parametrize(
    "existing", ["not json", '{"not": "a list"}'], ids=["corrupt", "not_list"]
)
def test_unusable_log_file_is_replaced(wrapper, log_path, image_path, existing):
    log_path.parent.mkdir()
    log_path.write_text(existing)

    wrapper.predict(None, image_path)

    assert len(_read_log(log_path)) == 1


def test_failed_log_write_does_not_break_the_prediction(wrapper, log_path, image_path):
    # A directory at the log path makes opening it fail
    log_path.mkdir(parents=True)

    assert wrapper.predict(None, image_path)["predictions"] == [1]


def test_failed_log_folder_creation_does_not_break_the_prediction(tmp_path, image_path):
    # A file where the log folder should be makes creating the folder fail
    (tmp_path / "blocked").write_text("not a folder")
    wrapper = SimpleLoggingWrapper(
        model=FixedProbabilityModel(), log_path=tmp_path / "blocked" / "inference_log.json"
    )

    assert wrapper.predict(None, image_path)["predictions"] == [1]


def test_log_folder_is_created_for_a_loaded_model(wrapper, log_path, image_path):
    # Unpickling skips __init__, which is what happens when a saved model is loaded
    loaded = pickle.loads(pickle.dumps(wrapper))
    assert not log_path.parent.exists()

    loaded.predict(None, image_path)

    assert len(_read_log(log_path)) == 1


# --- construction and loading ------------------------------------------------


def test_default_log_path_stays_relative():
    # An absolute path would pickle the training machine's path into the served model
    wrapper = SimpleLoggingWrapper(model=FixedProbabilityModel())

    assert wrapper.log_path == Path("artifacts/inference_log.json")


def test_creating_a_wrapper_does_not_create_the_log_folder(tmp_path):
    SimpleLoggingWrapper(model=FixedProbabilityModel())

    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    "make_checkpoint",
    [
        pytest.param(
            lambda state: {"model_state_dict": state, "architecture": "SimpleCNN"},
            id="full_checkpoint",
        ),
        pytest.param(lambda state: state, id="bare_state_dict"),
    ],
)
def test_model_is_loaded_from_a_checkpoint_on_first_predict(
    tmp_path, log_path, image_path, make_checkpoint
):
    source = SimpleCNN()
    checkpoint_path = tmp_path / "model.pth"
    torch.save(make_checkpoint(source.state_dict()), checkpoint_path)
    wrapper = SimpleLoggingWrapper(model_path=checkpoint_path, log_path=log_path)

    result = wrapper.predict(None, image_path)

    assert result["predictions"] in ([0], [1])
    assert not wrapper.get_raw_model().training
    loaded = wrapper.get_raw_model().state_dict()
    assert all(torch.equal(value, loaded[key]) for key, value in source.state_dict().items())


def test_load_context_keeps_a_model_it_already_has(wrapper):
    model = wrapper.get_raw_model()

    wrapper.load_context(None)

    assert wrapper.get_raw_model() is model
    assert not model.training


def test_missing_model_file_raises(tmp_path, log_path):
    wrapper = SimpleLoggingWrapper(model_path=tmp_path / "missing.pth", log_path=log_path)

    with pytest.raises(FileNotFoundError):
        wrapper.load_context(None)


def test_unknown_architecture_raises(tmp_path, log_path):
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"architecture": "OtherNet"}, checkpoint_path)
    wrapper = SimpleLoggingWrapper(model_path=checkpoint_path, log_path=log_path)

    with pytest.raises(ValueError, match="Unknown architecture"):
        wrapper.load_context(None)
