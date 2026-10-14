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
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, VectorParams, Distance

logger = logging.getLogger('uvicorn')
embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
logger.info(os.getenv('OLLAMA_HOST'))

_qdrant_client = None


def get_qdrant_client(path="data/my_qdrant_db"):
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=path)
    return _qdrant_client


def init_qdrant_collection(collection_name, vector_size=384, client=None):
    if client is None:
        client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    return client


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


def create_chat_function(create_system_prompt, qdrant_client, collection_name, mongo, model, name,
                         history_prompt):

    def query_chat_history(user_input, user_id, n_results=10):
        try:
            if hasattr(qdrant_client, 'count'):
                count_res = qdrant_client.count(collection_name=collection_name)
                count = count_res.count if hasattr(count_res, 'count') else count_res
            else:
                count = n_results
            if count == 0:
                return None
            actual_n = min(n_results, count) if count > 0 else n_results
            encoded = embedder.encode([user_input]).tolist()
            query_embeddings = encoded[0] if isinstance(encoded[0], list) else encoded

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key='user_id',
                        match=MatchValue(value=user_id)
                    )
                ]
            )
            query_res = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embeddings,
                limit=actual_n,
                query_filter=query_filter,
            )
            points = query_res.points if hasattr(query_res, 'points') else query_res
        except Exception as e:
            logger.error(f'Error querying qdrant: {e}')
            return None

        if not points:
            return None

        history_items = []
        for point in points:
            payload = point.payload if hasattr(point, 'payload') else (point.get('payload', {}) if isinstance(point, dict) else {})
            doc = payload.get('document', '')
            t = payload.get('time', '')
            if doc:
                history_items.append({'doc': doc, 'time': t})

        if not history_items:
            return None

        # Sort retrieved history chronologically so LLM reads history in temporal order
        history_items.sort(key=lambda x: x['time'])

        result = []
        for item in history_items:
            t = item['time']
            doc = item['doc']
            if t:
                result.append(f'{doc}\n\t\t-- {t}')
            else:
                result.append(doc)

        return '\n\n'.join(result) if result else None

    def chat(user_input, user_name, user_id):
        messages = []
        try:
            system = create_system_prompt(user_name, user_input)
        except TypeError:
            system = create_system_prompt(user_name)
        chat_history = query_chat_history(user_input, user_id)
        if chat_history:
            system += '\n\n' + history_prompt + '\n' + chat_history
        logger.info(system)
        messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': user_input})

        response = ollama.chat(model=model, messages=messages)
        reply = response['message']['content']
        logger.info(messages[-1])
        logger.info(reply)

        t = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        text = f'{user_name}: {user_input}\n{name}: {reply}'
        texts = [text]
        ids = [str(uuid.uuid4()) for _ in texts]

        embeddings = embedder.encode(texts).tolist()
        points = [
            PointStruct(
                id=doc_id,
                vector=emb,
                payload={
                    'document': doc_text,
                    'user_id': user_id,
                    'time': t,
                }
            )
            for doc_id, emb, doc_text in zip(ids, embeddings, texts)
        ]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
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
