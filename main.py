__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import mittens
import kiddos_bot
import yoshi
import pastor

app = FastAPI()

dir = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(dir, 'static')), name="static")

app.include_router(mittens.router)
app.include_router(kiddos_bot.router)
app.include_router(yoshi.router)
app.include_router(pastor.router)
