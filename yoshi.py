from datetime import datetime
import json
import requests
import logging

from fastapi import APIRouter
from fastapi import Request, HTTPException

import opencc
from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration, ReplyMessageRequest, TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import configparser
from pymongo import MongoClient

from utils import get_user_profile

logger = logging.getLogger('uvicorn')
config = configparser.ConfigParser()
config.read('config.ini')

# mittens
yoshi_channel_secret = config.get('yoshi', 'LINE_CHANNEL_SECRET')
yoshi_channel_access_token = config.get('yoshi', 'LINE_CHANNEL_ACCESS_TOKEN')
yoshi_configuration = Configuration(access_token=yoshi_channel_access_token)
yoshi_async_api_client = AsyncApiClient(yoshi_configuration)
yoshi_line_api = AsyncMessagingApi(yoshi_async_api_client)
yoshi_parser = WebhookParser(yoshi_channel_secret)

username = config.get('mongo', 'username')
password = config.get('mongo', 'password')

MODEL = 'qwen2'
HISTORY_SIZE = 30

mongo_url = f'mongodb://{username}:{password}@localhost/'
mongo_client = MongoClient(mongo_url)
db = mongo_client[MODEL]
mittens_collection = db['yoshi']

converter = opencc.OpenCC('s2t.json')

router = APIRouter()


def save_response(user_id, user_input, response):
  entry = {
    'model': MODEL,
    'user_id': user_id,
    'user_input': user_input,
    'response': response,
    't': datetime.now(),
  }
  mittens_collection.insert_many([entry])


def chat(messages):
  url = 'http://localhost:11434/api/chat'
  data = {
    'model': MODEL,
    'messages': messages,
    'stream': False,
    'options': {
      'repeat_last_n': 256,
      'temperature': 0.75,
      'num_gpu': 30,
    },
  }
  r = requests.post(url, json=data)
  r.raise_for_status()

  body = json.loads(r.content)
  if "error" in body:
    raise Exception(body["error"])
  message = body.get("message", {})
  return message


def create_system_prompt(user_name):
  return f"""你是吉野 (Yoshi)
你將跟使用者({user_name})對話
吉野 (Yoshi), 这个名字在日语里意味着“好”或“幸运”，但他本人却一点都不外向。他就像一个可爱的矛盾体。十岁的吉野身材瘦小，总像是撑不起那些稍微有点大的衣服。他柔软的深棕色头发总是被风吹得乱糟糟的，半遮住那双藏在超大眼镜後面，充满好奇的大眼睛。吉野经常脸红，每当他害羞时，可爱的粉红色就会爬上他的脖子，染红他的脸颊，让他看起来像一朵活生生的樱花。吉野的笑容虽然罕见，却令人着迷。当他笑起来的时候，嘴角会微微翘起，让他从一个害羞的紫罗兰，变成一朵阳光灿烂的向日葵，尽管有点紧张。
"""


def query_messages(user_id):
  cursor = mittens_collection.find({
    'user_id': user_id
  }, {}).sort('t', -1).limit(HISTORY_SIZE)
  messages = []
  query_result = []
  for entry in cursor:
    query_result.append(entry)

  for entry in query_result[::-1]:
    messages.append({
      'role': 'user',
      'content': entry['user_input'],
    })
    messages.append({'role': 'assistant', 'content': entry['response']})
  return messages


def run_chat(user_id, user_name, user_input):
  messages = query_messages(user_id)
  messages = messages + [{
    'role': 'system',
    'content': create_system_prompt(user_name)
  }]
  messages.append({'role': 'user', 'content': user_input})
  logger.info(str(messages[-1]))
  message = chat(messages)
  logger.info(str(message))
  save_response(user_id, user_input, message['content'])
  return message


@router.post('/yoshi')
async def mittens(request: Request):
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
    message = run_chat(user_id, user_name, user_input)
    content = converter.convert(message['content'].strip())

    request = ReplyMessageRequest(
      reply_token=event.reply_token,
      messages=[TextMessage(text=content)],
    )
    await yoshi_line_api.reply_message(request)
  return 'OK'
