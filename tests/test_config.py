"""Tests for config loading and validation."""
import json

import pytest

from verifier_probe import DEFAULT_CONFIG, load_config


def test_defaults_and_override():
    # Overriding one key must keep the others at their defaults.
    cfg = load_config({"seed": 5})
    assert cfg["seed"] == 5
    assert cfg["verifier_calls"] == DEFAULT_CONFIG["verifier_calls"]


def test_unknown_key_is_rejected():
    # A typo must not silently be ignored.
    with pytest.raises(ValueError):
        load_config({"sed": 1})


def test_bad_value_is_rejected():
    with pytest.raises(ValueError):
        load_config({"verifier_calls": 0})


def test_config_from_json_file(tmp_path):
    # Round-trip through a JSON file.
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 3}))
    assert load_config(path)["seed"] == 3