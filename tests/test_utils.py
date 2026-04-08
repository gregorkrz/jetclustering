"""Tests for utility modules: paths, import_tools."""

import os
import tempfile
import pytest

from src.utils.paths import get_path
from src.utils.import_tools import import_module


# ---------------------------------------------------------------------------
# get_path
# ---------------------------------------------------------------------------

class TestGetPath:
    def test_absolute_path_returned_as_is(self):
        result = get_path("/absolute/path/to/file", type="code")
        assert result == "/absolute/path/to/file"

    def test_code_path(self, monkeypatch):
        monkeypatch.setenv("SVJ_CODE_ROOT", "/home/user/code")
        result = get_path("config.yaml", type="code")
        assert result == "/home/user/code/config.yaml"

    def test_data_path(self, monkeypatch):
        monkeypatch.setenv("SVJ_DATA_ROOT", "/data")
        result = get_path("events.root", type="data")
        assert result == "/data/events.root"

    def test_preprocessed_data_path(self, monkeypatch):
        monkeypatch.setenv("SVJ_PREPROCESSED_DATA_ROOT", "/preproc")
        result = get_path("train.h5", type="preprocessed_data")
        assert result == "/preproc/train.h5"

    def test_results_path(self, monkeypatch):
        monkeypatch.setenv("SVJ_RESULTS_ROOT", "/results")
        result = get_path("output.pkl", type="results")
        assert result == "/results/output.pkl"

    def test_results_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SVJ_RESULTS_ROOT", str(tmp_path / "primary"))
        monkeypatch.setenv("SVJ_RESULTS_ROOT_FALLBACK", "/fallback")
        result = get_path("output.pkl", type="results", fallback=True)
        assert result == "/fallback/output.pkl"

    def test_results_no_fallback_when_exists(self, monkeypatch, tmp_path):
        primary = tmp_path / "primary"
        primary.mkdir()
        (primary / "output.pkl").touch()
        monkeypatch.setenv("SVJ_RESULTS_ROOT", str(primary))
        monkeypatch.setenv("SVJ_RESULTS_ROOT_FALLBACK", "/fallback")
        result = get_path("output.pkl", type="results", fallback=True)
        assert result == str(primary / "output.pkl")

    def test_invalid_type_raises(self):
        with pytest.raises(AssertionError):
            get_path("file.txt", type="invalid")

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("SVJ_CODE_ROOT", "/code")
        result = get_path("  config.yaml  ", type="code")
        assert result == "/code/config.yaml"


# ---------------------------------------------------------------------------
# import_module
# ---------------------------------------------------------------------------

class TestImportModule:
    def test_import_simple_module(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("MY_CONSTANT = 42\ndef my_func(): return 'hello'\n")
            f.flush()
            mod = import_module(f.name, name="test_mod")
            assert mod.MY_CONSTANT == 42
            assert mod.my_func() == "hello"
        os.unlink(f.name)

    def test_import_with_class(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class Foo:\n    x = 10\n")
            f.flush()
            mod = import_module(f.name)
            assert mod.Foo.x == 10
        os.unlink(f.name)
