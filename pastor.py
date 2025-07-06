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
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from bible import get_bible_chapters
from utils import get_user_profile

logger = logging.getLogger('uvicorn')
config = configparser.ConfigParser()
config.read('config.ini')

# pastor
pastor_channel_secret = config.get('pastor', 'LINE_CHANNEL_SECRET')
pastor_channel_access_token = config.get('pastor', 'LINE_CHANNEL_ACCESS_TOKEN')
pastor_configuration = Configuration(access_token=pastor_channel_access_token)
pastor_async_api_client = AsyncApiClient(pastor_configuration)
pastor_line_api = AsyncMessagingApi(pastor_async_api_client)
pastor_parser = WebhookParser(pastor_channel_secret)

username = config.get('mongo', 'username')
password = config.get('mongo', 'password')

# MODEL = 'qwen:7b'
MODEL = 'llama3'
HISTORY_SIZE = 30

mongo_url = f'mongodb://{username}:{password}@localhost/'
mongo_client = MongoClient(mongo_url)
db = mongo_client[MODEL]
pastor_collection = db['pastor']

embeddings = OllamaEmbeddings(model=MODEL)
chapters = get_bible_chapters()
documents = [Document(page_content=c) for c in chapters]
db = Chroma(persist_directory="./bible_chroma", embedding_function=embeddings)

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
  pastor_collection.insert_many([entry])


def chat(messages):
  url = 'http://localhost:11434/api/chat'
  data = {
    'model': MODEL,
    'messages': messages,
    'stream': False,
    'options': {
      'repeat_last_n': 256,
      'temperature': 0.95,
      'num_predict': 1024,
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


def query_messages(user_id):
  cursor = pastor_collection.find({
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
  # docs = db.similarity_search(user_input)
  #
  # context = []
  # for doc in docs:
  #   context.append({'role': 'assistant', 'content': doc.page_content})
  #   print(context[-1])

  messages = query_messages(user_id)
  messages = [{
    'role': 'system',
    'content': create_system_prompt(user_name)
  }] + messages
  # messages += context
  user_input += '\n請用中文回答'
  messages.append({'role': 'user', 'content': user_input})
  logger.info(str(messages[-1]))

  message = chat(messages)
  logger.info(str(message))
  save_response(user_id, user_input, message['content'])
  return message


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

    user_id = event.source.user_id
    user_input = event.message.text
    profile = get_user_profile(user_id, pastor_channel_access_token)
    user_name = profile['displayName']
    message = run_chat(user_id, user_name, user_input)
    content = converter.convert(message['content'].strip())

    request = ReplyMessageRequest(
      reply_token=event.reply_token,
      messages=[TextMessage(text=content)],
    )
    await pastor_line_api.reply_message(request)
  return 'OK'
