from pathlib import Path

import pytest

from yolov8.config import Settings, load_settings


def test_default_settings_are_valid() -> None:
    Settings().validate()


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Settings(confidence=1.5).validate()


def test_load_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("epochs: 5\nconfidence: 0.4\n", encoding="utf-8")
    settings = load_settings(path)
    assert settings.epochs == 5
    assert settings.confidence == 0.4
