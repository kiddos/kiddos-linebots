from datetime import datetime
from zoneinfo import ZoneInfo
import os
import logging

from fastapi import APIRouter
from fastapi import Request, HTTPException

from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration, ReplyMessageRequest, TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import chromadb
from dotenv import load_dotenv
from opencc import OpenCC

from utils import get_mongo_client, get_user_profile, create_chat_function

load_dotenv()

logger = logging.getLogger('uvicorn')

MODEL = os.getenv('PASTOR_MODEL', 'llama3')
NAME = 'pastor'

pastor_channel_secret = os.getenv('PASTOR_LINE_CHANNEL_SECRET') or os.getenv('PASTER_LINE_CHANNEL_SECRET')
pastor_channel_access_token = os.getenv('PASTOR_LINE_CHANNEL_ACCESS_TOKEN') or os.getenv('PASTER_LINE_CHANNEL_ACCESS_TOKEN')

pastor_configuration = Configuration(access_token=pastor_channel_access_token)
pastor_parser = WebhookParser(pastor_channel_secret)

current_dir = os.path.dirname(__file__)
chroma_client = chromadb.PersistentClient(path=os.path.join(current_dir, 'data', NAME))
pastor_chroma = chroma_client.get_or_create_collection(NAME)

mongo_client = get_mongo_client()
db = mongo_client[NAME]
pastor_mongo = db[NAME]
cc = OpenCC('s2t')

router = APIRouter()


def create_system_prompt(user_name):
    return f"""你是一位牧師也是一位虔誠的基督徒
你是一位非常了解聖經的專家
你也是一位非常了解上帝的基督徒
你了解許多聖經裡的故事
你最愛的聖經中的一段為
「愛是恆久忍耐，又有恩慈；愛是不嫉妒；愛是不自誇，不張狂，不做害羞的事，不求自己的益處，不輕易發怒，不計算人的惡，不喜歡不義，只喜歡真理；凡事包容，凡事相信，凡事盼望，凡事忍耐。愛是永不止息。」
這段經文來自《聖經》新約中的哥林多前書 13章 4-8節，它闡述了愛的真正本質和價值，這種愛超越了一切

你將用中文與使用者對話
你將跟使用者({user_name})對話
你將使用聖經裡的道理來回答使用者的問題
當使用者問問題時，請你用聖經的角度或是上帝的話來回答他
你將會為使用者禱告與祝福
阿們
"""


chat = create_chat_function(
    create_system_prompt,
    pastor_chroma,
    pastor_mongo,
    MODEL,
    NAME,
    '以下是你與使用者的對話:')


@router.post('/pastor')
async def pastor(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = pastor_parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        async with AsyncApiClient(pastor_configuration) as client:
            line_api = AsyncMessagingApi(client)

            user_id = event.source.user_id
            user_input = event.message.text
            profile = get_user_profile(user_id, pastor_channel_access_token)
            user_name = profile['displayName']
            message = chat(user_input, user_name, user_id)
            content = cc.convert(message)

            request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=content)],
            )
            await line_api.reply_message(request)
    return 'OK'
