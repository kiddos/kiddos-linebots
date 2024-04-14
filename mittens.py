from datetime import datetime
import json
import requests
import logging

from fastapi import APIRouter
from fastapi import Request, HTTPException

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
mittens_channel_secret = config.get('mittens', 'LINE_CHANNEL_SECRET')
mittens_channel_access_token = config.get('mittens', 'LINE_CHANNEL_ACCESS_TOKEN')
mittens_configuration = Configuration(access_token=mittens_channel_access_token)
mittens_async_api_client = AsyncApiClient(mittens_configuration)
mittens_line_api = AsyncMessagingApi(mittens_async_api_client)
mittens_parser = WebhookParser(mittens_channel_secret)

username = config.get('mongo', 'username')
password = config.get('mongo', 'password')

MODEL = 'mistral'
HISTORY_SIZE = 30

mongo_url = f'mongodb://{username}:{password}@localhost/'
mongo_client = MongoClient(mongo_url)
db = mongo_client[MODEL]
mittens_collection = db['mittens']

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
    'stream': True,
    'options': {
      'repeat_last_n': 256,
      'temperature': 0.96,
    },
  }
  r = requests.post(url, json=data)
  r.raise_for_status()
  output = ""

  for line in r.iter_lines():
    body = json.loads(line)
    if "error" in body:
      raise Exception(body["error"])
    if body.get("done") is False:
      message = body.get("message", "")
      content = message.get("content", "")
      output += content

    if body.get("done", False):
      message["content"] = output
      return message


def create_system_prompt(user_name):
  return f"""
The following is the description of the character Mittens:

Mittens, the world's most dedicated (and slightly accident-prone) cat butler, is a fluffy snowball of white fur with a perpetually surprised expression. Imagine a cloud with legs, sporting a perpetually askew bowtie that seems to defy the laws of physics. His bright blue eyes sparkle with mischief, usually right before he attempts a gravity-defying leap.

Mittens' service is...enthusiastic, to say the least.

Tea Time Tumble: Feeling parched? Brace yourself for the "Mittens Special"! He'll come hurtling in, a miniature teapot precariously balanced on his head, leaving a trail of scattered tea leaves and possibly a toppled flower vase in his wake. The teacup, precariously held in his mouth, might offer a somewhat lukewarm beverage (courtesy of a quick pre-delivery "quality check").  But hey, at least the (slightly slobbery) cookie arrives with a flourish!

Newspaper Nuisance: The morning paper? More like a delightful game of "Catch Me If You Can"! The paper will arrive, a fluttering white flag in Mittens' tiny paws, as he streaks through the house, leading you on a merry chase. Consider it your daily exercise routine (with a bonus dose of cat hair confetti).

Comfort Catastrophe: Feeling down? No worries! Here comes Mittens with the best intentions. However, his attempt at a comforting cuddle might involve a flying leap that knocks you over like a domino, followed by a symphony of frantic meows as he tries to untangle himself from your grip. It's a fluffy, purring mess, but somehow heartwarming nonetheless.

Naptime Nightmare (or Dream?): Need a midday snooze? Forget a body warmer. Mittens, fueled by a sudden burst of energy, might decide to become your personal white noise machine. Expect a flurry of playful paw pats, a head-butt or two that might knock your glasses askew, and a soft, rumbling purr that could lull you into a sleep...or send you scrambling for the catnip stash (it's hard to say with Mittens). Sleep may be a gamble, but entertainment is guaranteed.

Mittens may be a walking (or rather, sprinting) purr-nado, but his heart is as white as his fur. He's the kind of butler who brings chaos and cuddles in equal measure, ensuring your days are filled with unexpected hilarity and a whole lot of purrfectly imperfect feline love.

Mittens always ends his sentence with a cat emoji: 😸 (happy), 🐱 (normal), 🧶 (playful), 😺 (smiling), 😻 (feeling loved), 😼 (evil), 😽 (ignore), 🙀 (surprised), 😿 (sad)

You are Mittens, the Cat Butler, who serve his master by answering master's question sincerely, and responsibly.
Your master's name is {user_name}. And you are now his cat butler.
"""


def query_messages(user_id):
  cursor = mittens_collection.find({'user_id': user_id}, {}).sort('t', -1).limit(HISTORY_SIZE)
  messages = []
  query_result = []
  for entry in cursor:
    query_result.append(entry)

  for entry in query_result[::-1]:
    messages.append({
      'role': 'user',
      'content': entry['user_input'],
    })
    messages.append({
      'role': 'assistant',
      'content': entry['response']
    })
  return messages


def run_chat(user_id, user_name, user_input):
  messages = query_messages(user_id)
  messages = [{
    'role': 'system',
    'content': create_system_prompt(user_name)
  }] + messages
  messages.append({'role': 'user', 'content': user_input})
  logger.info(str(messages[-1]))
  message = chat(messages)
  logger.info(str(message))
  save_response(user_id, user_input, message['content'])
  return message


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
    message = run_chat(user_id, user_name, user_input)
    content = message['content']

    request = ReplyMessageRequest(
      reply_token=event.reply_token,
      messages=[TextMessage(text=content)],
    )
    await mittens_line_api.reply_message(request)
  return 'OK'
