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
from dotenv import load_dotenv
from opencc import OpenCC

from utils import get_mongo_client, get_user_profile, create_chat_function, init_qdrant_collection, embedder

load_dotenv()

logger = logging.getLogger('uvicorn')

MODEL = os.getenv('PASTOR_MODEL', 'llama3')
USE_GEMINI_API = (os.getenv('PASTOR_USE_GEMINI_API') or os.getenv('USE_GEMINI_API') or 'false').lower() == 'true'
NAME = 'pastor'
BIBLE_COLLECTION = 'bible'

pastor_channel_secret = os.getenv('PASTOR_LINE_CHANNEL_SECRET') or os.getenv('PASTER_LINE_CHANNEL_SECRET')
pastor_channel_access_token = os.getenv('PASTOR_LINE_CHANNEL_ACCESS_TOKEN') or os.getenv('PASTER_LINE_CHANNEL_ACCESS_TOKEN')

pastor_configuration = Configuration(access_token=pastor_channel_access_token)
pastor_parser = WebhookParser(pastor_channel_secret)

current_dir = os.path.dirname(__file__)
qdrant_client = init_qdrant_collection(NAME)

mongo_client = get_mongo_client()
db = mongo_client[NAME]
pastor_mongo = db[NAME]
cc = OpenCC('s2t')

router = APIRouter()


def search_bible(query: str, limit: int = 3) -> str:
    if not query:
        return ""
    try:
        if not qdrant_client.collection_exists(BIBLE_COLLECTION):
            return ""
        encoded = embedder.encode([query]).tolist()
        query_vector = encoded[0] if isinstance(encoded[0], list) else encoded
        query_res = qdrant_client.query_points(
            collection_name=BIBLE_COLLECTION,
            query=query_vector,
            limit=limit,
        )
        points = query_res.points if hasattr(query_res, 'points') else query_res
        if not points:
            return ""

        verses = []
        for point in points:
            payload = point.payload if hasattr(point, 'payload') else (point.get('payload', {}) if isinstance(point, dict) else {})
            # LangChain QdrantVectorStore stores content in 'page_content' or custom payload field
            content = payload.get('page_content') or payload.get('document', '')
            if content:
                verses.append(content)
        return "\n\n".join(verses)
    except Exception as e:
        logger.error(f"Error querying Bible vector db: {e}")
        return ""


def create_system_prompt(user_name, user_input=None):
    prompt = f"""你是一位牧師也是一位虔誠的基督徒
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
    if user_input:
        bible_context = search_bible(user_input)
        if bible_context:
            prompt += f"\n\n以下是相關的聖經經文參考:\n{bible_context}"
    return prompt


chat = create_chat_function(
    create_system_prompt,
    qdrant_client,
    NAME,
    pastor_mongo,
    MODEL,
    NAME,
    '以下是你與使用者的對話:',
    use_gemini_api=USE_GEMINI_API)


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
