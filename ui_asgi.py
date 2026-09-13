"""Launch: .venv/bin/uvicorn ui_asgi:app --host 127.0.0.1 --port 8767."""
from contextlib import asynccontextmanager
from pathlib import Path
import marimo
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
import ui_server

@asynccontextmanager
async def lifespan(app):
    ui_server.start_worker()
    yield

app=FastAPI(lifespan=lifespan)
notebooks=marimo.create_asgi_app().with_app(path='',root=str(Path(__file__).parent/'ui/notebook.py')).build()
app.mount('/notebook',notebooks)
app.mount('/',WSGIMiddleware(ui_server.app))
