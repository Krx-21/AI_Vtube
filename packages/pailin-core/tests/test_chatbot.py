"""Tests for the chatbot module."""

import pytest
from unittest.mock import MagicMock, patch

from pailin_core.ai.chatbot import Chatbot


class TestChatbot:
    """Tests for the Chatbot class."""

    @patch("pailin_core.ai.chatbot.genai")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    def test_init_configures_api(self, mock_genai: MagicMock) -> None:
        mock_chat = MagicMock()
        mock_genai.GenerativeModel.return_value.start_chat.return_value = mock_chat

        bot = Chatbot()

        mock_genai.configure.assert_called_once_with(api_key="test-key")
        assert bot.chat is mock_chat

    @patch("pailin_core.ai.chatbot.genai")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_chat_with_gemini_returns_filtered_text(
        self, mock_genai: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = "สวัสดี! ไพลินค่า!"
        mock_chat = MagicMock()
        mock_chat.send_message_async = MagicMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value.start_chat.return_value = mock_chat

        bot = Chatbot()
        result = await bot.chat_with_gemini("สวัสดี")

        assert "!" not in result
        assert "ไพลิน" in result

    @patch("pailin_core.ai.chatbot.genai")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_chat_with_gemini_handles_error(self, mock_genai: MagicMock) -> None:
        mock_chat = MagicMock()
        mock_chat.send_message_async.side_effect = RuntimeError("API error")
        mock_genai.GenerativeModel.return_value.start_chat.return_value = mock_chat

        bot = Chatbot()
        result = await bot.chat_with_gemini("test")

        assert "ไพลิน" in result  # Should return a Pailin-style error

    @patch("pailin_core.ai.chatbot.genai")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    def test_reset_conversation(self, mock_genai: MagicMock) -> None:
        mock_chat = MagicMock()
        mock_genai.GenerativeModel.return_value.start_chat.return_value = mock_chat

        bot = Chatbot()
        bot.reset_conversation()

        # start_chat should have been called twice: init + reset
        assert mock_genai.GenerativeModel.return_value.start_chat.call_count == 2

    def test_init_raises_without_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                Chatbot()
