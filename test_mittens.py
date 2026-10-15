import os
# Provide dummy env vars for LINE channel secrets during imports/testing if not set
os.environ.setdefault('MITTENS_LINE_CHANNEL_SECRET', 'dummy_secret')
os.environ.setdefault('MITTENS_LINE_CHANNEL_ACCESS_TOKEN', 'dummy_token')
os.environ.setdefault('YOSHI_LINE_CHANNEL_SECRET', 'dummy_secret')
os.environ.setdefault('YOSHI_LINE_CHANNEL_ACCESS_TOKEN', 'dummy_token')
os.environ.setdefault('PASTOR_LINE_CHANNEL_SECRET', 'dummy_secret')
os.environ.setdefault('PASTOR_LINE_CHANNEL_ACCESS_TOKEN', 'dummy_token')
os.environ.setdefault('KIDDOS_BOT_LINE_CHANNEL_SECRET', 'dummy_secret')
os.environ.setdefault('KIDDOS_BOT_LINE_CHANNEL_ACCESS_TOKEN', 'dummy_token')

import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from fastapi.testclient import TestClient
from linebot.v3.webhooks import MessageEvent, TextMessageContent, UserSource, DeliveryContext

import mittens
import main


class TestMittensMemory(unittest.TestCase):

    def setUp(self):
        self.temp_db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db_file.close()
        self.db_path = self.temp_db_file.name
        mittens.init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_sqlite_db_operations(self):
        user_id = "user123"

        # Initial empty
        memories = mittens.get_user_memories(user_id, db_path=self.db_path)
        self.assertEqual(memories, {})

        # Insert favorite food
        mittens.set_user_memory(user_id, "favorite food", "cookie", db_path=self.db_path)
        memories = mittens.get_user_memories(user_id, db_path=self.db_path)
        self.assertEqual(memories, {"favorite food": "cookie"})

        # Update favorite food
        mittens.set_user_memory(user_id, "favorite food", "chocolate", db_path=self.db_path)
        memories = mittens.get_user_memories(user_id, db_path=self.db_path)
        self.assertEqual(memories, {"favorite food": "chocolate"})

        # Add another key
        mittens.set_user_memory(user_id, "pet name", "Felix", db_path=self.db_path)
        memories = mittens.get_user_memories(user_id, db_path=self.db_path)
        self.assertEqual(memories, {"favorite food": "chocolate", "pet name": "Felix"})

        # Delete key
        mittens.delete_user_memory(user_id, "favorite food", db_path=self.db_path)
        memories = mittens.get_user_memories(user_id, db_path=self.db_path)
        self.assertEqual(memories, {"pet name": "Felix"})

    @patch("mittens.get_user_memories")
    def test_create_system_prompt_injects_memories(self, mock_get_memories):
        mock_get_memories.return_value = {"favorite food": "cookie", "pet name": "Felix"}

        prompt = mittens.create_system_prompt("Alice", user_id="user123")

        self.assertIn("Your master's name is Alice", prompt)
        self.assertIn("Here is what you remember about your master:", prompt)
        self.assertIn("- favorite food: cookie", prompt)
        self.assertIn("- pet name: Felix", prompt)

    @patch("mittens.generate_ollama_response")
    def test_determine_and_update_memory_llm(self, mock_ollama_resp):
        mock_ollama_resp.return_value = '```json\n{"updates": [{"action": "set", "key": "favorite food", "value": "cookie"}]}\n```'

        user_id = "user_test_llm"
        mittens.determine_and_update_memory(
            user_id, "My favorite food is cookie", "Meow! Cookies are delicious!", "Bob", db_path=self.db_path
        )

        memories = mittens.get_user_memories(user_id, db_path=self.db_path)
        self.assertEqual(memories, {"favorite food": "cookie"})

    @patch("mittens.determine_and_update_memory")
    @patch("mittens._chat_fn")
    def test_chat_wrapper_calls_determine_and_update_memory(self, mock_chat_fn, mock_determine):
        mock_chat_fn.return_value = "Meow! Pure perfection."

        res = mittens.chat("I love tuna", "Alice", "user_alice")

        self.assertEqual(res, "Meow! Pure perfection.")
        mock_chat_fn.assert_called_once_with("I love tuna", "Alice", "user_alice")
        mock_determine.assert_called_once_with("user_alice", "I love tuna", "Meow! Pure perfection.", "Alice")

    @patch("mittens.chat")
    @patch("mittens.get_user_profile")
    @patch("mittens.AsyncApiClient")
    @patch("mittens.AsyncMessagingApi")
    @patch("mittens.mittens_parser")
    def test_mittens_route_valid_event(
        self, mock_parser, mock_messaging_api, mock_api_client_cls, mock_get_profile, mock_chat
    ):
        client = TestClient(main.app)
        event = MessageEvent(
            replyToken='token_mittens',
            source=UserSource(userId='user_mittens'),
            message=TextMessageContent(id='msg_m1', text='I like fish', quoteToken='dummy_quote'),
            timestamp=1234567890,
            mode='active',
            webhookEventId='evt_m1',
            deliveryContext=DeliveryContext(isRedelivery=False)
        )

        mock_parser.parse.return_value = [event]
        mock_get_profile.return_value = {'displayName': 'MasterAlice'}
        mock_chat.return_value = 'Meow! Fish is great.'

        mock_api_instance = AsyncMock()
        mock_messaging_api.return_value = mock_api_instance

        mock_client_ctx = AsyncMock()
        mock_api_client_cls.return_value.__aenter__.return_value = mock_client_ctx

        headers = {'X-Line-Signature': 'dummy_sig'}
        response = client.post('/mittens', data='{}', headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), 'OK')
        mock_chat.assert_called_once_with('I like fish', 'MasterAlice', 'user_mittens')
        mock_api_instance.reply_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
