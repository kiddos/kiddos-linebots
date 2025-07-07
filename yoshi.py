from datetime import datetime
from zoneinfo import ZoneInfo
import os

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

MODEL = os.getenv('YOSHI_MODEL')
NAME = 'yoshi'

yoshi_channel_secret = os.getenv('YOSHI_LINE_CHANNEL_SECRET')
yoshi_channel_access_token = os.getenv('YOSHI_LINE_CHANNEL_ACCESS_TOKEN')

yoshi_configuration = Configuration(access_token=yoshi_channel_access_token)
yoshi_async_api_client = AsyncApiClient(yoshi_configuration)
yoshi_line_api = AsyncMessagingApi(yoshi_async_api_client)
yoshi_parser = WebhookParser(yoshi_channel_secret)

current_dir = os.path.dirname(__file__)
# client = chromadb.PersistentClient(path=os.path.join(current_dir, 'data', NAME))
chroma_client = chromadb.HttpClient(host=os.getenv('CHROMA_HOST'), port=int(os.getenv('CHROMA_PORT')))
yoshi_chroma = chroma_client.get_or_create_collection(NAME)

mongo_client = get_mongo_client()
db = mongo_client[NAME]
yoshi_mongo = db[NAME]
cc = OpenCC('s2t')

router = APIRouter()


with open(os.path.join(current_dir, 'yoshi.txt'), 'r') as f:
    yoshi_prompt = f.read()


def create_system_prompt(user_name):
    now = datetime.now(ZoneInfo('Asia/Taipei'))
    t = now.strftime('%Y-%m-%d %H:%M:%S')
    p = yoshi_prompt
    p += f'\n現在時間為 {t}。'
    p += f"\n你將跟 {user_name} 對話。\n"
    return p


chat = create_chat_function(
    create_system_prompt,
    yoshi_chroma,
    yoshi_mongo,
    MODEL,
    NAME,
    '以下是你與使用者的對話:')


@router.post('/yoshi')
async def yoshi(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = yoshi_parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        user_id = event.source.user_id
        user_input = event.message.text
        profile = get_user_profile(user_id, yoshi_channel_access_token)
        user_name = profile['displayName']
        message = chat(user_input, user_name, user_id)
        content = cc.convert(message)

        request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=content)],
        )
        await yoshi_line_api.reply_message(request)
    return 'OK'
