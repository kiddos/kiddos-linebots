from datetime import datetime
from zoneinfo import ZoneInfo
import os
import logging
import requests

from fastapi import APIRouter, Request, HTTPException

from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration, ReplyMessageRequest, TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from dotenv import load_dotenv
from tavily import TavilyClient

from utils import get_mongo_client, get_user_profile, create_chat_function, init_qdrant_collection

load_dotenv()

logger = logging.getLogger('uvicorn')

MODEL = os.getenv('KIDDOS_BOT_MODEL') or os.getenv('KIDDOS_MODEL') or 'gemma3'
USE_GEMINI_API = (os.getenv('KIDDOS_BOT_USE_GEMINI_API') or os.getenv('KIDDOS_USE_GEMINI_API') or os.getenv('USE_GEMINI_API') or 'false').lower() == 'true'
NAME = 'kiddos'

bot_channel_secret = os.getenv('KIDDOS_BOT_LINE_CHANNEL_SECRET') or os.getenv('KIDDOS_LINE_CHANNEL_SECRET')
bot_channel_access_token = os.getenv('KIDDOS_BOT_LINE_CHANNEL_ACCESS_TOKEN') or os.getenv('KIDDOS_LINE_CHANNEL_ACCESS_TOKEN')

bot_configuration = Configuration(access_token=bot_channel_access_token)
bot_parser = WebhookParser(bot_channel_secret)

current_dir = os.path.dirname(__file__)
qdrant_client = init_qdrant_collection(NAME)

mongo_client = get_mongo_client()
db = mongo_client[NAME]
kiddos_mongo = db[NAME]

router = APIRouter()


def search_tavily(query: str) -> str:
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        return ""
    try:
        tavily_client = TavilyClient(api_key=api_key)
        response = tavily_client.search(query=query)
        results = response.get('results', [])
        if not results:
            return ""
        context_items = []
        for item in results:
            title = item.get('title', '')
            content = item.get('content', '')
            if content:
                context_items.append(f"- {title}: {content}")
        return "\n".join(context_items)
    except Exception as e:
        logger.error(f"Error querying Tavily API: {e}")
        return ""


def control_ledstrip(brightness: int, mode: str) -> str:
    """Control the ESP32 LED strip brightness and mode.

    Args:
        brightness: Brightness level from 0 to 100.
        mode: Lighting mode ('white', 'rgb', or 'random').
    """
    url = os.getenv("LEDSTRIP_URL", "http://192.168.0.136/api/ledstrip")
    try:
        brightness_val = int(brightness)
        brightness_val = max(0, min(100, brightness_val))
    except (ValueError, TypeError):
        return f"Invalid brightness value '{brightness}'. Brightness must be an integer between 0 and 100."

    mode_str = str(mode).lower() if mode else ""
    if mode_str not in ["white", "rgb", "random"]:
        return f"Invalid mode '{mode}'. Mode must be 'white', 'rgb', or 'random'."

    payload = {
        "brightness": brightness_val,
        "mode": mode_str
    }
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return f"Successfully set LED strip brightness to {brightness_val} and mode to '{mode_str}'."
        else:
            return f"LED strip controller returned status code {res.status_code}: {res.text}"
    except Exception as e:
        logger.error(f"Error controlling LED strip: {e}")
        return f"Failed to control LED strip: {e}"


def create_system_prompt(user_name, user_input=None):
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    t = now.strftime("%Y-%m-%d %H:%M:%S")
    p = f"You are a helpful AI assistant who answers questions from {user_name}."
    p += f"\nCurrently the time is {t}."
    p += "\nYou have access to a tool to control an ESP32 LED strip. You can set its brightness (0-100) and mode ('white', 'rgb', 'random') when requested by the user."
    if user_input:
        search_results = search_tavily(user_input)
        if search_results:
            p += f"\n\nHere is information from web search about the user's question:\n{search_results}"
    return p


chat = create_chat_function(
    create_system_prompt,
    qdrant_client,
    NAME,
    kiddos_mongo,
    MODEL,
    NAME,
    'The following is the chat history between you and the user:',
    use_gemini_api=USE_GEMINI_API,
    tools=[control_ledstrip])


@router.post('/kiddos-bot')
@router.post('/kiddos')
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = bot_parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        async with AsyncApiClient(bot_configuration) as client:
            line_api = AsyncMessagingApi(client)

            user_id = event.source.user_id
            user_input = event.message.text
            profile = get_user_profile(user_id, bot_channel_access_token)
            user_name = profile['displayName']
            message = chat(user_input, user_name, user_id)
            content = message

            request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=content)],
            )
            await line_api.reply_message(request)
    return 'OK'
