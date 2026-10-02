"""Regression tests for delimiter-aware loading of user-uploaded text files.

A semicolon/tab/pipe export must not silently collapse into a single column:
``pd.read_csv`` does not raise for those, it just returns every line as one
field, which then propagates a one-column dataset through the whole pipeline.
Conversely a genuinely single-column CSV must keep its single column.
"""
import pandas as pd
import pytest

from app.utils.file_utils import detect_delimiter, load_dataframe


def _write(tmp_path, text: str, name: str = "data.csv"):
    path = tmp_path / name
    # newline="" so the csv module and pandas see identical line endings.
    path.write_text(text, newline="", encoding="utf-8")
    return path


class TestDetectDelimiter:
    def test_detects_semicolon(self, tmp_path):
        p = _write(tmp_path, "a;b;c\n1;2;3\n4;5;6\n")
        assert detect_delimiter(p) == ";"

    def test_detects_tab(self, tmp_path):
        p = _write(tmp_path, "a\tb\tc\n1\t2\t3\n")
        assert detect_delimiter(p) == "\t"

    def test_detects_pipe(self, tmp_path):
        p = _write(tmp_path, "a|b|c\n1|2|3\n")
        assert detect_delimiter(p) == "|"

    def test_detects_comma(self, tmp_path):
        p = _write(tmp_path, "a,b,c\n1,2,3\n")
        assert detect_delimiter(p) == ","

    def test_single_column_has_no_delimiter(self, tmp_path):
        p = _write(tmp_path, "name\nalpha\nbeta\n")
        assert detect_delimiter(p) is None

    def test_empty_file_has_no_delimiter(self, tmp_path):
        p = _write(tmp_path, "")
        assert detect_delimiter(p) is None


class TestLoadDataframeDelimiter:
    def test_semicolon_file_is_not_one_column(self, tmp_path):
        p = _write(tmp_path, "a;b;c\n1;2;3\n4;5;6\n")
        df = load_dataframe(p)
        assert list(df.columns) == ["a", "b", "c"]
        assert df.shape == (2, 3)

    def test_tab_file_is_not_one_column(self, tmp_path):
        p = _write(tmp_path, "a\tb\n1\t2\n")
        df = load_dataframe(p)
        assert list(df.columns) == ["a", "b"]

    def test_european_decimal_commas_survive(self, tmp_path):
        """A semicolon file with comma decimals must not be split on commas."""
        p = _write(tmp_path, "a;b\n1,5;2,5\n3,5;4,5\n")
        df = load_dataframe(p)
        assert list(df.columns) == ["a", "b"]
        assert df["a"].tolist() == ["1,5", "3,5"]

    def test_genuine_single_column_stays_single_column(self, tmp_path):
        p = _write(tmp_path, "name\nalpha\nbeta\ngamma\n")
        df = load_dataframe(p)
        assert df.shape == (3, 1)
        assert list(df.columns) == ["name"]

    def test_ragged_commas_match_plain_pandas(self, tmp_path):
        """Ambiguous input is deferred to pandas, never reinterpreted.

        When commas appear in only some lines the file is genuinely ragged and
        pandas promotes the first field to an index. The loader must return
        exactly what pandas would return rather than inventing a delimiter.
        """
        text = "note\nhello, world\nfoo bar\nqux, quux\ncorge grault\n"
        p = _write(tmp_path, text)
        assert detect_delimiter(p) is None
        pd.testing.assert_frame_equal(load_dataframe(p), pd.read_csv(p))

    def test_comma_csv_is_unchanged(self, tmp_path):
        p = _write(tmp_path, "a,b,c\n1,2,3\n4,5,6\n")
        assert load_dataframe(p).shape == (2, 3)

    def test_nrows_zero_still_reports_columns(self, tmp_path):
        """The /columns endpoint reads a header-only frame; it must see the
        real column count for a semicolon file too."""
        p = _write(tmp_path, "a;b;c\n1;2;3\n4;5;6\n")
        assert list(load_dataframe(p, nrows=0).columns) == ["a", "b", "c"]

    def test_empty_file_still_raises(self, tmp_path):
        with pytest.raises(pd.errors.EmptyDataError):
            load_dataframe(_write(tmp_path, ""))
