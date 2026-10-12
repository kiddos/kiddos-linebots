import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock sentence_transformers and SentenceTransformer before importing utils
mock_st = MagicMock()
mock_embedder_instance = MagicMock()
mock_st.SentenceTransformer.return_value = mock_embedder_instance
sys.modules['sentence_transformers'] = mock_st

import utils


class TestChatHistory(unittest.TestCase):

    @patch('utils.ollama_client')
    def test_query_chat_history(self, mock_ollama_client):
        mock_embedder_instance.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

        mock_chroma = MagicMock()
        mock_chroma.count.return_value = 5
        mock_chroma.query.return_value = {
            'documents': [['User: Hi\nBot: Hello', 'User: Bye\nBot: See ya']],
            'metadatas': [[
                {'user_id': 'user123', 'time': '2025-01-01 10:00:00'},
                {'user_id': 'user123', 'time': '2025-01-01 10:05:00'},
            ]]
        }

        mock_mongo = MagicMock()

        def dummy_prompt(user_name):
            return f"System prompt for {user_name}"

        chat_fn = utils.create_chat_function(
            dummy_prompt,
            mock_chroma,
            mock_mongo,
            'test_model',
            'test_bot',
            'History:'
        )

        mock_ollama_client.chat.return_value = {
            'message': {'content': 'How can I help you?'}
        }

        response = chat_fn('What is up?', 'Alice', 'user123')

        mock_embedder_instance.encode.assert_any_call(['What is up?'])

        mock_chroma.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=5,
            where={'user_id': 'user123'}
        )

        mock_mongo.insert_many.assert_called_once()
        self.assertEqual(response, 'How can I help you?')

    @patch('utils.ollama_client')
    def test_query_chat_history_chronological_sorting(self, mock_ollama_client):
        mock_embedder_instance.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

        mock_chroma = MagicMock()
        mock_chroma.count.return_value = 2
        # Vector search returns documents out of temporal order (e.g. higher vector similarity first)
        mock_chroma.query.return_value = {
            'documents': [['User: Later message\nBot: Reply 2', 'User: Earlier message\nBot: Reply 1']],
            'metadatas': [[
                {'user_id': 'user123', 'time': '2025-01-01 12:00:00'},
                {'user_id': 'user123', 'time': '2025-01-01 10:00:00'},
            ]]
        }

        mock_mongo = MagicMock()

        def dummy_prompt(user_name):
            return "System prompt"

        chat_fn = utils.create_chat_function(
            dummy_prompt,
            mock_chroma,
            mock_mongo,
            'test_model',
            'test_bot',
            'History:'
        )

        mock_ollama_client.chat.return_value = {
            'message': {'content': 'Response'}
        }

        chat_fn('Test query', 'Bob', 'user123')

        call_args = mock_ollama_client.chat.call_args[1]
        system_msg = call_args['messages'][0]['content']

        pos_earlier = system_msg.find('Earlier message')
        pos_later = system_msg.find('Later message')

        self.assertNotEqual(pos_earlier, -1)
        self.assertNotEqual(pos_later, -1)
        self.assertLess(pos_earlier, pos_later)


if __name__ == '__main__':
    unittest.main()
