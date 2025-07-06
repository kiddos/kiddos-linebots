import os
from fastapi import FastAPI
import mittens
import yoshi
# import pastor

app = FastAPI()

dir = os.path.dirname(__file__)
app.include_router(mittens.router)
app.include_router(yoshi.router)
# app.include_router(pastor.router)
