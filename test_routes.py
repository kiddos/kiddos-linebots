import os
# Provide dummy env vars for LINE channel secrets during imports/testing if not set
os.environ.setdefault('MITTENS_LINE_CHANNEL_SECRET', 'dummy_secret')
os.environ.setdefault('MITTENS_LINE_CHANNEL_ACCESS_TOKEN', 'dummy_token')
os.environ.setdefault('YOSHI_LINE_CHANNEL_SECRET', 'dummy_secret')
os.environ.setdefault('YOSHI_LINE_CHANNEL_ACCESS_TOKEN', 'dummy_token')
os.environ.setdefault('PASTOR_LINE_CHANNEL_SECRET', 'dummy_secret')
os.environ.setdefault('PASTOR_LINE_CHANNEL_ACCESS_TOKEN', 'dummy_token')

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


if __name__ == '__main__':
    unittest.main()
