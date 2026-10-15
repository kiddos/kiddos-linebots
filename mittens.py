from datetime import datetime
from zoneinfo import ZoneInfo
import os
import logging
import json
import sqlite3

from fastapi import APIRouter
from fastapi import Request, HTTPException

from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration, ReplyMessageRequest, TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from dotenv import load_dotenv

from utils import (
    get_mongo_client,
    get_user_profile,
    create_chat_function,
    init_qdrant_collection,
    generate_gemini_response,
    generate_ollama_response,
)

load_dotenv()

logger = logging.getLogger('uvicorn')

# mittens
mittens_channel_secret = os.getenv('MITTENS_LINE_CHANNEL_SECRET')
mittens_channel_access_token = os.getenv('MITTENS_LINE_CHANNEL_ACCESS_TOKEN')

mittens_configuration = Configuration(
    access_token=mittens_channel_access_token)

MODEL = os.getenv('MITTENS_MODEL')
USE_GEMINI_API = (os.getenv('MITTENS_USE_GEMINI_API') or os.getenv('USE_GEMINI_API') or 'false').lower() == 'true'
NAME = 'mittens'

current_dir = os.path.dirname(__file__)
qdrant_client = init_qdrant_collection(NAME)

mongo_client = get_mongo_client()
db = mongo_client[NAME]
mittens_mongo = db[NAME]
mittens_parser = WebhookParser(mittens_channel_secret)

DB_PATH = os.getenv("MITTENS_SQLITE_PATH", os.path.join(current_dir, "data", "mittens_memory.db"))


def get_db_connection(db_path=DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, key)
                );
            """)
    finally:
        conn.close()


init_db()


def get_user_memories(user_id, db_path=DB_PATH):
    if not user_id:
        return {}
    conn = get_db_connection(db_path)
    try:
        rows = conn.execute("SELECT key, value FROM user_memory WHERE user_id = ?", (user_id,)).fetchall()
        return {row["key"]: row["value"] for row in rows}
    finally:
        conn.close()


def set_user_memory(user_id, key, value, db_path=DB_PATH):
    if not user_id or not key:
        return
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                INSERT INTO user_memory (user_id, key, value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, key, value))
    finally:
        conn.close()


def delete_user_memory(user_id, key, db_path=DB_PATH):
    if not user_id or not key:
        return
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM user_memory WHERE user_id = ? AND key = ?", (user_id, key))
    finally:
        conn.close()


def determine_and_update_memory(user_id, user_input, response, user_name, db_path=DB_PATH):
    if not user_id:
        return
    memories = get_user_memories(user_id, db_path=db_path)
    memories_str = "\n".join([f"- {k}: {v}" for k, v in memories.items()]) if memories else "None"

    system_prompt = (
        "You are a long-term memory extraction system for a chat assistant.\n"
        "Your task is to analyze the user message and assistant reply to identify any new key facts about the user (master) "
        "that should be saved to long-term memory, updated, or removed.\n"
        "Key facts are attributes/preferences/details about the user (e.g., favorite food, name of pet, birthday, job, likes/dislikes).\n"
        "Return ONLY a valid JSON object in the following format:\n"
        '{"updates": [{"action": "set", "key": "favorite food", "value": "cookie"}]}\n'
        'Use "action": "set" to add or update a key-value, or "action": "delete" to remove a key.\n'
        'If no user key-value facts are learned or modified in this message, return {"updates": []}.\n'
        "Do not include any commentary or extra text outside the JSON object."
    )

    prompt = (
        f"Master's name: {user_name}\n"
        f"Currently stored memories for user:\n{memories_str}\n\n"
        f"Latest Interaction:\n"
        f"Master: {user_input}\n"
        f"Mittens: {response}\n\n"
        "JSON output:"
    )

    try:
        if USE_GEMINI_API:
            llm_res = generate_gemini_response(MODEL, prompt, system_prompt)
        else:
            llm_res = generate_ollama_response(MODEL, prompt, system_prompt)

        if not llm_res:
            return

        clean_res = llm_res.strip()
        if clean_res.startswith("```"):
            lines = clean_res.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_res = "\n".join(lines).strip()

        data = json.loads(clean_res)
        updates = data.get("updates", [])
        for item in updates:
            action = item.get("action", "set").lower()
            key = item.get("key")
            value = item.get("value")
            if action == "set" and key and value:
                set_user_memory(user_id, key, value, db_path=db_path)
            elif action == "delete" and key:
                delete_user_memory(user_id, key, db_path=db_path)
    except Exception as e:
        logger.error(f"Error determining/updating Mittens memory for user {user_id}: {e}")


router = APIRouter()

with open(os.path.join(current_dir, 'mittens.txt'), 'r') as f:
    mittens_prompt = f.read()


def create_system_prompt(user_name, user_input=None, user_id=None):
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    t = now.strftime("%Y-%m-%d %H:%M:%S")
    p = mittens_prompt
    p += f"\nYour master's name is {user_name}. And you are now the cat butler."
    p += f"\nIt is {t} right now, and it's time to serve your master.\n"
    if user_id:
        memories = get_user_memories(user_id)
        if memories:
            p += "\nHere is what you remember about your master:\n"
            for k, v in memories.items():
                p += f"- {k}: {v}\n"
    return p


_chat_fn = create_chat_function(
    create_system_prompt,
    qdrant_client,
    NAME,
    mittens_mongo,
    MODEL,
    NAME,
    'The following is the chat history between you and your master:',
    use_gemini_api=USE_GEMINI_API)


def chat(user_input, user_name, user_id):
    reply = _chat_fn(user_input, user_name, user_id)
    try:
        determine_and_update_memory(user_id, user_input, reply, user_name)
    except Exception as e:
        logger.error(f"Failed to update memory for user {user_id}: {e}")
    return reply


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

        async with AsyncApiClient(mittens_configuration) as client:
            line_api = AsyncMessagingApi(client)

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
            await line_api.reply_message(request)
    return 'OK'
