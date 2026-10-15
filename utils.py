import os
import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import uuid

from pymongo import MongoClient
import ollama
from google import genai
from google.genai import types
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


def generate_gemini_response(model, user_input, system, tools=None):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_input),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        system_instruction=system,
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        ),
        audio_transcription_config=types.AudioTranscriptionConfig(),
        tools=tools,
    )
    if tools:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        return response.text or ""
    else:
        chunks = []
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if text := chunk.text:
                chunks.append(text)
        return "".join(chunks)


def generate_ollama_response(model, user_input, system, tools=None):
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user_input}
    ]
    kwargs = {}
    if tools:
        kwargs['tools'] = tools

    response = ollama.chat(model=model, messages=messages, **kwargs)
    logger.info(messages[-1])

    msg = response.get('message') if isinstance(response, dict) else getattr(response, 'message', {})
    tool_calls = msg.get('tool_calls') if isinstance(msg, dict) else getattr(msg, 'tool_calls', None)

    if tool_calls:
        tool_map = {fn.__name__: fn for fn in tools} if tools else {}
        messages.append(msg)
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                fn_name = tool_call.get('function', {}).get('name')
                fn_args = tool_call.get('function', {}).get('arguments', {})
            else:
                fn_name = getattr(getattr(tool_call, 'function', None), 'name', None)
                fn_args = getattr(getattr(tool_call, 'function', None), 'arguments', {})

            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    pass

            if fn_name in tool_map:
                try:
                    result = tool_map[fn_name](**fn_args)
                except Exception as e:
                    result = f"Error executing tool {fn_name}: {e}"
            else:
                result = f"Tool {fn_name} not found."

            messages.append({
                'role': 'tool',
                'content': str(result),
            })

        response = ollama.chat(model=model, messages=messages, **kwargs)
        final_msg = response.get('message') if isinstance(response, dict) else getattr(response, 'message', {})
        if isinstance(final_msg, dict):
            return final_msg.get('content', '')
        return getattr(final_msg, 'content', '')

    if isinstance(msg, dict):
        return msg.get('content', '')
    return getattr(msg, 'content', '')


def create_chat_function(create_system_prompt, qdrant_client, collection_name, mongo, model, name,
                         history_prompt, use_gemini_api=False, tools=None):

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
        import inspect
        sig = inspect.signature(create_system_prompt)
        params = sig.parameters
        kwargs = {}
        if 'user_name' in params:
            kwargs['user_name'] = user_name
        if 'user_input' in params:
            kwargs['user_input'] = user_input
        if 'user_id' in params:
            kwargs['user_id'] = user_id

        try:
            system = create_system_prompt(**kwargs)
        except TypeError:
            try:
                system = create_system_prompt(user_name, user_input)
            except TypeError:
                system = create_system_prompt(user_name)
        chat_history = query_chat_history(user_input, user_id)
        if chat_history:
            system += '\n\n' + history_prompt + '\n' + chat_history
        logger.info(system)

        if use_gemini_api:
            reply = generate_gemini_response(model, user_input, system, tools=tools)
        else:
            reply = generate_ollama_response(model, user_input, system, tools=tools)

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
