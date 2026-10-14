import unittest
from unittest.mock import MagicMock, patch

import utils


class TestChatHistory(unittest.TestCase):

    @patch('utils.embedder')
    @patch('utils.ollama')
    def test_query_chat_history(self, mock_ollama, mock_embedder):
        mock_embedder.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

        mock_qdrant = MagicMock()
        mock_qdrant.count.return_value.count = 5

        mock_p1 = MagicMock()
        mock_p1.payload = {'document': 'User: Hi\nBot: Hello', 'user_id': 'user123', 'time': '2025-01-01 10:00:00'}
        mock_p2 = MagicMock()
        mock_p2.payload = {'document': 'User: Bye\nBot: See ya', 'user_id': 'user123', 'time': '2025-01-01 10:05:00'}
        mock_qdrant.query_points.return_value.points = [mock_p1, mock_p2]

        mock_mongo = MagicMock()

        def dummy_prompt(user_name):
            return f"System prompt for {user_name}"

        chat_fn = utils.create_chat_function(
            dummy_prompt,
            mock_qdrant,
            'test_bot',
            mock_mongo,
            'test_model',
            'test_bot',
            'History:'
        )

        mock_ollama.chat.return_value = {
            'message': {'content': 'How can I help you?'}
        }

        response = chat_fn('What is up?', 'Alice', 'user123')

        mock_embedder.encode.assert_any_call(['What is up?'])

        mock_qdrant.query_points.assert_called_once()
        _, kwargs = mock_qdrant.query_points.call_args
        self.assertEqual(kwargs['collection_name'], 'test_bot')
        self.assertEqual(kwargs['query'], [0.1, 0.2, 0.3])
        self.assertEqual(kwargs['limit'], 5)

        mock_qdrant.upsert.assert_called_once()
        mock_mongo.insert_many.assert_called_once()
        self.assertEqual(response, 'How can I help you?')

    @patch('utils.embedder')
    @patch('utils.ollama')
    def test_query_chat_history_chronological_sorting(self, mock_ollama, mock_embedder):
        mock_embedder.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

        mock_qdrant = MagicMock()
        mock_qdrant.count.return_value.count = 2

        # Vector search returns documents out of temporal order (e.g. higher vector similarity first)
        mock_p1 = MagicMock()
        mock_p1.payload = {'document': 'User: Later message\nBot: Reply 2', 'user_id': 'user123', 'time': '2025-01-01 12:00:00'}
        mock_p2 = MagicMock()
        mock_p2.payload = {'document': 'User: Earlier message\nBot: Reply 1', 'user_id': 'user123', 'time': '2025-01-01 10:00:00'}
        mock_qdrant.query_points.return_value.points = [mock_p1, mock_p2]

        mock_mongo = MagicMock()

        def dummy_prompt(user_name):
            return "System prompt"

        chat_fn = utils.create_chat_function(
            dummy_prompt,
            mock_qdrant,
            'test_bot',
            mock_mongo,
            'test_model',
            'test_bot',
            'History:'
        )

        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        chat_fn('Test query', 'Bob', 'user123')

        call_args = mock_ollama.chat.call_args[1]
        system_msg = call_args['messages'][0]['content']

        pos_earlier = system_msg.find('Earlier message')
        pos_later = system_msg.find('Later message')

        self.assertNotEqual(pos_earlier, -1)
        self.assertNotEqual(pos_later, -1)
        self.assertLess(pos_earlier, pos_later)


if __name__ == '__main__':
    unittest.main()
