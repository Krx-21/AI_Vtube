"""Tests for text sanitizer module."""

import pytest

from pailin_core.text.sanitizer import remove_special_characters


class TestRemoveSpecialCharacters:
    """Tests for the remove_special_characters function."""

    def test_removes_punctuation(self) -> None:
        assert remove_special_characters("Hello, world!") == "Hello world"

    def test_removes_symbols(self) -> None:
        assert remove_special_characters("Price: $100 @ 50%") == "Price 100  50"

    def test_preserves_thai_text(self) -> None:
        text = "สวัสดีครับ ไพลินค่ะ"
        assert remove_special_characters(text) == "สวัสดีครับ ไพลินค่ะ"

    def test_removes_special_symbols_from_thai(self) -> None:
        text = "ฮายยย~! ไพลินเองจ้า!"
        result = remove_special_characters(text)
        assert "!" not in result
        assert "~" not in result
        assert "ไพลิน" in result

    def test_empty_string(self) -> None:
        assert remove_special_characters("") == ""

    def test_only_special_characters(self) -> None:
        assert remove_special_characters("!@#$%^&*()") == ""

    def test_preserves_spaces(self) -> None:
        assert "  " in remove_special_characters("a  b")

    def test_preserves_digits(self) -> None:
        assert remove_special_characters("abc123") == "abc123"

    def test_unicode_symbols_removed(self) -> None:
        assert remove_special_characters("❤♥★") == ""

    def test_mixed_content(self) -> None:
        text = "ไพลิน ❤ says: Hello!"
        result = remove_special_characters(text)
        assert "ไพลิน" in result
        assert "Hello" in result
        assert "❤" not in result
        assert "!" not in result
