from datetime import datetime
import os
import logging
import uuid

from sentence_transformers import SentenceTransformer
import ollama
from fastapi import APIRouter
from fastapi import Request, HTTPException
from pymongo import MongoClient

from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration, ReplyMessageRequest, TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import configparser
import chromadb

from utils import get_user_profile

logger = logging.getLogger('uvicorn')
config = configparser.ConfigParser()
config.read('config.ini')

username = config.get('mongo', 'username')
password = config.get('mongo', 'password')

# mittens
mittens_channel_secret = config.get('mittens', 'LINE_CHANNEL_SECRET')
mittens_channel_access_token = config.get('mittens',
                                          'LINE_CHANNEL_ACCESS_TOKEN')
mittens_configuration = Configuration(
    access_token=mittens_channel_access_token)
mittens_async_api_client = AsyncApiClient(mittens_configuration)
mittens_line_api = AsyncMessagingApi(mittens_async_api_client)
mittens_parser = WebhookParser(mittens_channel_secret)

MODEL = 'gemma3'
HISTORY_SIZE = 10
NAME = 'mittens'

embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
current_dir = os.path.dirname(__file__)
client = chromadb.PersistentClient(path=os.path.join(current_dir, NAME))
mittens_chroma = client.get_or_create_collection(NAME)

mongo_url = f'mongodb://{username}:{password}@localhost/'
mongo_client = MongoClient(mongo_url)
db = mongo_client[NAME]
mittens_mongo = db['mittens']

router = APIRouter()


def create_system_prompt(user_name):
    now = datetime.now()
    t = now.strftime("%Y-%m-%d %H:%M:%S")
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
Your master's name is {user_name}. And you are now his cat butler. It is {t} right now, and it's time to serve your master.
"""


def query_chat_history(user_input, user_id):
    results = mittens_chroma.query(
        query_texts=[user_input],
        n_results=6,
        where={'user_id': user_id}
    )
    documents = [doc[0] for doc in results['documents'] if len(doc) > 0]
    if len(documents) == 0:
        return None
    return '\n'.join(documents)


def chat(user_input, user_name, user_id):
    messages = []
    system = create_system_prompt(user_name)
    chat_history = query_chat_history(user_input, user_id)
    if chat_history:
        system += '\n\nThe following is the chat history between you and your master:\n' + chat_history
    messages.append({'role': 'system', 'content': system})

    cursor = mittens_mongo.find({'user_id': user_id}, {}).sort('t', -1).limit(HISTORY_SIZE)
    recent_history = []
    for entry in cursor:
        recent_history.append({
            'role': 'assistant',
            'content': entry['response']
        })
        recent_history.append({
            'role': 'user',
            'content': entry['user_input'],
        })
    recent_history = recent_history[::-1]
    messages += recent_history

    messages.append({'role': 'user', 'content': user_input})

    response = ollama.chat(model=MODEL, messages=messages)
    reply = response['message']['content']
    logger.info(messages[-1])
    logger.info(reply)

    t = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    chat_history = f'time: {t}\nUser: {user_input}\nMittens: {reply}'
    texts = [chat_history]
    ids = [str(uuid.uuid4()) for _ in texts]

    embeddings = embedder.encode(texts).tolist()
    mittens_chroma.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{'user_id': user_id}],
        ids=ids
    )
    entry = {
        'model': MODEL,
        'user_id': user_id,
        'user_input': user_input,
        'response': reply,
        't': datetime.now(),
    }
    mittens_mongo.insert_many([entry])
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
