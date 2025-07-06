from datetime import datetime
import os

from fastapi import APIRouter
from fastapi import Request, HTTPException

from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration, ReplyMessageRequest, TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import chromadb
from dotenv import load_dotenv

from utils import get_mongo_client, get_user_profile, create_chat_function

load_dotenv()

# mittens
mittens_channel_secret = os.getenv('MITTENS_LINE_CHANNEL_SECRET')
mittens_channel_access_token = os.getenv('MITTENS_LINE_CHANNEL_ACCESS_TOKEN')

mittens_configuration = Configuration(
    access_token=mittens_channel_access_token)
mittens_async_api_client = AsyncApiClient(mittens_configuration)
mittens_line_api = AsyncMessagingApi(mittens_async_api_client)
mittens_parser = WebhookParser(mittens_channel_secret)

MODEL = os.getenv('MITTENS_MODEL')
NAME = 'mittens'

current_dir = os.path.dirname(__file__)
client = chromadb.PersistentClient(path=os.path.join(current_dir, 'data', NAME))
mittens_chroma = client.get_or_create_collection(NAME)

mongo_client = get_mongo_client()
db = mongo_client[NAME]
mittens_mongo = db[NAME]

router = APIRouter()

with open(os.path.join(current_dir, 'mittens.txt'), 'r') as f:
    mittens_prompt = f.read()


def create_system_prompt(user_name):
    now = datetime.now()
    t = now.strftime("%Y-%m-%d %H:%M:%S")
    p = mittens_prompt
    p += f"\nYour master's name is {user_name}. And you are now the cat butler."
    p += f"\nIt is {t} right now, and it's time to serve your master.\n"
    return p


chat = create_chat_function(
    create_system_prompt,
    mittens_chroma,
    mittens_mongo,
    MODEL,
    NAME,
    'The following is the chat history between you and your master:')


@router.post('/mittens')
async def mittens(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = mittens_parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        user_id = event.source.user_id
        user_input = event.message.text
        profile = get_user_profile(user_id, mittens_channel_access_token)
        user_name = profile['displayName']
        message = chat(user_input, user_name, user_id)
        content = message

        request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=content)],
        )
        await mittens_line_api.reply_message(request)
    return 'OK'
