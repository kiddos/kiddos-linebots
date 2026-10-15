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

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from linebot.v3.webhooks import MessageEvent, TextMessageContent, UserSource, DeliveryContext

import main


class TestRoutes(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(main.app)

    @patch('yoshi.chat')
    @patch('yoshi.get_user_profile')
    @patch('yoshi.AsyncApiClient')
    @patch('yoshi.AsyncMessagingApi')
    @patch('yoshi.yoshi_parser')
    def test_yoshi_route_valid_event(
        self, mock_parser, mock_messaging_api, mock_api_client_cls, mock_get_profile, mock_chat
    ):
        event = MessageEvent(
            replyToken='token_yoshi',
            source=UserSource(userId='user_yoshi'),
            message=TextMessageContent(id='msg_1', text='Hello Yoshi', quoteToken='dummy_quote'),
            timestamp=1234567890,
            mode='active',
            webhookEventId='evt_1',
            deliveryContext=DeliveryContext(isRedelivery=False)
        )

        mock_parser.parse.return_value = [event]
        mock_get_profile.return_value = {'displayName': 'TestUser'}
        mock_chat.return_value = 'Yoshi response'

        mock_api_instance = AsyncMock()
        mock_messaging_api.return_value = mock_api_instance

        mock_client_ctx = AsyncMock()
        mock_api_client_cls.return_value.__aenter__.return_value = mock_client_ctx

        headers = {'X-Line-Signature': 'dummy_sig'}
        response = self.client.post('/yoshi', data='{}', headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), 'OK')
        mock_chat.assert_called_once_with('Hello Yoshi', 'TestUser', 'user_yoshi')
        mock_api_instance.reply_message.assert_called_once()

    @patch('pastor.chat')
    @patch('pastor.get_user_profile')
    @patch('pastor.AsyncApiClient')
    @patch('pastor.AsyncMessagingApi')
    @patch('pastor.pastor_parser')
    def test_pastor_route_valid_event(
        self, mock_parser, mock_messaging_api, mock_api_client_cls, mock_get_profile, mock_chat
    ):
        event = MessageEvent(
            replyToken='token_pastor',
            source=UserSource(userId='user_pastor'),
            message=TextMessageContent(id='msg_2', text='Hello Pastor', quoteToken='dummy_quote'),
            timestamp=1234567890,
            mode='active',
            webhookEventId='evt_2',
            deliveryContext=DeliveryContext(isRedelivery=False)
        )

        mock_parser.parse.return_value = [event]
        mock_get_profile.return_value = {'displayName': 'TestUser'}
        mock_chat.return_value = 'Pastor response'

        mock_api_instance = AsyncMock()
        mock_messaging_api.return_value = mock_api_instance

        mock_client_ctx = AsyncMock()
        mock_api_client_cls.return_value.__aenter__.return_value = mock_client_ctx

        headers = {'X-Line-Signature': 'dummy_sig'}
        response = self.client.post('/pastor', data='{}', headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), 'OK')
        mock_chat.assert_called_once_with('Hello Pastor', 'TestUser', 'user_pastor')
        mock_api_instance.reply_message.assert_called_once()

    @patch('kiddos_bot.chat')
    @patch('kiddos_bot.get_user_profile')
    @patch('kiddos_bot.AsyncApiClient')
    @patch('kiddos_bot.AsyncMessagingApi')
    @patch('kiddos_bot.bot_parser')
    def test_kiddos_bot_route_valid_event(
        self, mock_parser, mock_messaging_api, mock_api_client_cls, mock_get_profile, mock_chat
    ):
        event = MessageEvent(
            replyToken='token_kiddos',
            source=UserSource(userId='user_kiddos'),
            message=TextMessageContent(id='msg_3', text='What is the capital of France?', quoteToken='dummy_quote'),
            timestamp=1234567890,
            mode='active',
            webhookEventId='evt_3',
            deliveryContext=DeliveryContext(isRedelivery=False)
        )

        mock_parser.parse.return_value = [event]
        mock_get_profile.return_value = {'displayName': 'TestUser'}
        mock_chat.return_value = 'The capital of France is Paris.'

        mock_api_instance = AsyncMock()
        mock_messaging_api.return_value = mock_api_instance

        mock_client_ctx = AsyncMock()
        mock_api_client_cls.return_value.__aenter__.return_value = mock_client_ctx

        headers = {'X-Line-Signature': 'dummy_sig'}
        response = self.client.post('/kiddos-bot', data='{}', headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), 'OK')
        mock_chat.assert_called_once_with('What is the capital of France?', 'TestUser', 'user_kiddos')
        mock_api_instance.reply_message.assert_called_once()

    @patch('kiddos_bot.TavilyClient')
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'fake_key'})
    def test_search_tavily(self, mock_tavily_client_cls):
        import kiddos_bot
        mock_client = MagicMock()
        mock_tavily_client_cls.return_value = mock_client
        mock_client.search.return_value = {
            'results': [
                {'title': 'Paris Info', 'content': 'Paris is the capital of France.'}
            ]
        }

        res = kiddos_bot.search_tavily('capital of France')
        self.assertIn('Paris is the capital of France.', res)
        mock_client.search.assert_called_once_with(query='capital of France')

    @patch('kiddos_bot.requests.post')
    @patch.dict(os.environ, {'LED_CONTROLLER_URL': 'http://192.168.1.200'})
    def test_control_ledstrip_custom_env_url(self, mock_post):
        import kiddos_bot
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        result = kiddos_bot.control_ledstrip(30, 'random')
        self.assertIn('Successfully set LED strip brightness to 30 and mode to \'random\'', result)
        mock_post.assert_called_once_with(
            'http://192.168.1.200/api/ledstrip',
            json={'brightness': 30, 'mode': 'random'},
            timeout=5
        )

    @patch('kiddos_bot.requests.post')
    @patch.dict(os.environ, {'LED_CONTROLLER_URL': 'http://192.168.1.200'})
    def test_control_ledstrip_success(self, mock_post):
        import kiddos_bot
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        result = kiddos_bot.control_ledstrip(50, 'white')
        self.assertIn('Successfully set LED strip brightness to 50 and mode to \'white\'', result)
        mock_post.assert_called_once_with(
            'http://192.168.1.200/api/ledstrip',
            json={'brightness': 50, 'mode': 'white'},
            timeout=5
        )

    def test_control_ledstrip_invalid_mode(self):
        import kiddos_bot
        result = kiddos_bot.control_ledstrip(50, 'invalid_mode')
        self.assertIn("Invalid mode 'invalid_mode'", result)

    @patch('kiddos_bot.requests.post')
    @patch.dict(os.environ, {'LED_CONTROLLER_URL': 'http://192.168.1.200'})
    def test_control_ledstrip_brightness_clamping(self, mock_post):
        import kiddos_bot
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = kiddos_bot.control_ledstrip(150, 'rgb')
        self.assertIn('Successfully set LED strip brightness to 100 and mode to \'rgb\'', result)
        mock_post.assert_called_once_with(
            'http://192.168.1.200/api/ledstrip',
            json={'brightness': 100, 'mode': 'rgb'},
            timeout=5
        )

    @patch('kiddos_bot.requests.post')
    def test_control_ledstrip_http_error(self, mock_post):
        import kiddos_bot
        mock_post.side_effect = Exception('Connection refused')

        result = kiddos_bot.control_ledstrip(50, 'random')
        self.assertIn('Failed to control LED strip: Connection refused', result)

    @patch('pastor.embedder')
    @patch('pastor.qdrant_client')
    def test_pastor_search_bible_and_prompt(self, mock_qdrant, mock_embedder):
        import pastor
        mock_qdrant.collection_exists.return_value = True
        mock_embedder.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

        mock_p1 = MagicMock()
        mock_p1.payload = {'page_content': '[創世紀 1:1 (GEN 1:1)]\n中文: 起初，上帝創造天地。'}
        mock_qdrant.query_points.return_value.points = [mock_p1]

        res = pastor.search_bible('創世')
        self.assertIn('起初，上帝創造天地。', res)
        mock_qdrant.query_points.assert_called_once()

        prompt = pastor.create_system_prompt('Alice', '創世')
        self.assertIn('以下是相關的聖經經文參考:', prompt)
        self.assertIn('起初，上帝創造天地。', prompt)


if __name__ == '__main__':
    unittest.main()
