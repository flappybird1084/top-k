"""Same-origin API and authenticated, read-only marimo run notebook."""
from pathlib import Path
import marimo
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.responses import RedirectResponse
import app as api

class NotebookAuth:
    def __init__(self, app): self.app=app
    async def __call__(self, scope, receive, send):
        if scope['type'] in ('http','websocket') and scope.get('path','').startswith('/notebook'):
            headers={k.decode().lower():v.decode() for k,v in scope['headers']}
            env={'REMOTE_ADDR':scope.get('client',('',0))[0], 'HTTP_COOKIE':headers.get('cookie',''),
                 'HTTP_ORIGIN':headers.get('origin',''),'HTTP_HOST':headers.get('host',''),
                 'wsgi.url_scheme':'https' if scope.get('scheme') in ('https','wss') else 'http'}
            if not api.authorized(env) or (scope['type']=='websocket' and not api.origin_ok(env)):
                if scope['type']=='websocket': await send({'type':'websocket.close','code':1008})
                else: await RedirectResponse('/login.html')(scope,receive,send)
                return
            scope['user']={'is_authenticated':True,'username':'workspace'}
        await self.app(scope,receive,send)

app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
app.add_middleware(NotebookAuth)
notebook=marimo.create_asgi_app(include_code=False).with_app(path='',root=str(Path(__file__).with_name('notebook.py'))).build()
app.mount('/notebook',notebook)
app.mount('/',WSGIMiddleware(api.application))
