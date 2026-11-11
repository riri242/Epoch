"""Smoke-test the Streamlit app using published report fallbacks."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_generated_outputs():
    app = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    view = AppTest.from_file(str(app)).run(timeout=45)
    assert not view.exception
    assert view.title[0].value == "Epoch"
    assert len(view.tabs) == 5
