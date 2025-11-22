import os
import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import uuid

from pymongo import MongoClient
import ollama
from sentence_transformers import SentenceTransformer

logger = logging.getLogger('uvicorn')
embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
ollama_client = ollama.Client()


def get_mongo_client():
    mongo_host = os.getenv('MONGO_HOST')
    mongo_username = os.getenv('MONGO_USERNAME')
    mongo_password = os.getenv('MONGO_PASSWORD')
    mongo_url = f'mongodb://{mongo_username}:{mongo_password}@{mongo_host}/'
    return MongoClient(mongo_url)


def get_user_profile(user_id, channel_access_token):
    url = f'https://api.line.me/v2/bot/profile/{user_id}'
    headers = {'Authorization': f'Bearer {channel_access_token}'}
    r = requests.get(url, headers=headers)
    return json.loads(r.content)


def create_chat_function(create_system_prompt, chroma, mongo, model, name,
                         history_prompt):

    def query_chat_history(user_input, user_id, n_results=10):
        results = chroma.query(
            query_texts=[user_input],
            n_results=n_results,
            where={'user_id': user_id},
        )
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        result = []
        for i, doc in enumerate(documents):
            metadata = metadatas[i]
            t = metadata['time']
            result.append(f'{doc}\n\t\t-- {t}')
        if len(result) == 0:
            return None
        print(result)
        return '\n\n'.join(result)

    def chat(user_input, user_name, user_id):
        messages = []
        system = create_system_prompt(user_name)
        chat_history = query_chat_history(user_input, user_id)
        if chat_history:
            system += '\n\n' + history_prompt + '\n' + chat_history
        logger.info(system)
        messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': user_input})

        response = ollama_client.chat(model=model, messages=messages)
        reply = response['message']['content']
        logger.info(messages[-1])
        logger.info(reply)

        t = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        text = f'{user_name}: {user_input}\n{name}: {reply}'
        texts = [text]
        ids = [str(uuid.uuid4()) for _ in texts]

        embeddings = embedder.encode(texts).tolist()
        chroma.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                'user_id': user_id,
                'time': t,
            }],
            ids=ids,
        )
        entry = {
            'model': model,
            'user_id': user_id,
            'user_input': user_input,
            'response': reply,
            't': datetime.now(ZoneInfo("Asia/Taipei")),
        }
        mongo.insert_many([entry])
        return reply

    return chat
