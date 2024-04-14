import os
import logging
from datetime import datetime
import requests
import json

from fastapi import APIRouter
from fastapi import Request, HTTPException

from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration, ReplyMessageRequest, TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import configparser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.agents.agent import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

MODEL = 'gemma'


@tool
def calculator_tool(query: str) -> str:
  """Perform math operations.
  Example:
    Query: 3102 * 48102
    this tool will return 149212404
  """
  try:
    return eval(query)
  except Exception:
    return ''


@tool
def current_time_tool() -> str:
  """Get current time"""
  t = datetime.now()
  return t.strftime("%Y-%m-%d, %H:%M:%S")


@tool
def conversation_tool(query: str) -> str:
  """this tool can answer conversational question from user"""
  url = 'http://localhost:11434/api/generate'
  system = "You are a helpful AI assistant who answer user's question."
  data = {
    'model': MODEL,
    'system': system,
    'prompt': query,
    'options': {
      'temperature': 0.96,
    },
  }

  r = requests.post(url, json=data)
  r.raise_for_status()
  output = ""

  try:
    for line in r.iter_lines():
      body = json.loads(line)
      if "error" in body:
        raise Exception(body["error"])
      if body.get("done") is False:
        message = body.get("message", "")
        content = message.get("content", "")
        output += content
    logger.info(f'conversation output: {output}')
    return output
  except Exception:
    return "I don't know the answer"


logger = logging.getLogger('uvicorn')
config = configparser.ConfigParser()
config.read('config.ini')

# bot
bot_channel_secret = config.get('kiddos-bot', 'LINE_CHANNEL_SECRET')
bot_channel_access_token = config.get('kiddos-bot', 'LINE_CHANNEL_ACCESS_TOKEN')
bot_configuration = Configuration(access_token=bot_channel_access_token)
bot_async_api_client = AsyncApiClient(bot_configuration)
bot_line_api = AsyncMessagingApi(bot_async_api_client)
bot_parser = WebhookParser(bot_channel_secret)

os.environ['TAVILY_API_KEY'] = config.get('travily', 'TAVILY_API_KEY')
travily_search_tool = TavilySearchResults()
python_repl_tool = PythonREPLTool()
api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
os.environ['GOOGLE_CSE_ID'] = config.get('google', 'GOOGLE_CSE_ID')
os.environ['GOOGLE_API_KEY'] = config.get('google', 'GOOGLE_API_KEY')

google_search = GoogleSearchAPIWrapper()

google_search_tool = Tool(
  name="google_search",
  description="Search Google for recent results.",
  func=google_search.run,
)

llm = Ollama(model=MODEL, temperature=0.1)
tools = [conversation_tool, calculator_tool, python_repl_tool, wiki_tool, google_search_tool]

template = """[INST] Answer the following questions as best you can. You have access to the following tools:

{tools}

To use a tool, please use the following format:
Thought: you should always think about what to do
Action: the tool to use, should be one of [{tool_names}]
Action Input: the input to the tool you selected
Observation: the result of the tool
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT NOTE:
1. There should always be a "Action:" after "Thought:"
2. There should always be a "Action Input:" after "Action:"
3. There should always be a "Observation:" after "Action:"

The following is an example.

Instruction: What is the date of today?
Thought: To get the date for today, I can use the Python_REPL tool to get the current date
Action: Action: Python_REPL
Action Input: import datetime; print(datetime.date.today())
2024-04-07
Observation: I now know the current date, which is April 7, 2024.
Final Answer: Today's date is April 7, 2024.


Instruction: {input}
Thought: {agent_scratchpad}
[/INST]
"""
prompt = ChatPromptTemplate.from_template(template)

tool_names = [tool.name for tool in tools]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
  agent=agent,
  tools=tools,
  handle_parsing_errors=True,
  verbose=True,
)

router = APIRouter()


@router.post("/kiddos-bot")
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

    user_input = event.message.text
    result = agent_executor.invoke({'input': user_input})
    message = TextMessage(text=result['output'])
    request = ReplyMessageRequest(reply_token=event.reply_token, messages=[message])
    await bot_line_api.reply_message(request)
  return 'OK'
