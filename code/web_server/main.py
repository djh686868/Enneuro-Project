import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from web_server.routers import models, datasets, training, explain

app = FastAPI(title="EnNeuro Web API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(explain.router)


@app.get("/ping")
def ping():
    return {"status": "ok"}


# 挂载前端静态文件（挂载在 /app 路径避免与 /api 冲突）
_static = os.path.join(os.path.dirname(__file__), "../../frontend_static")
_dist   = os.path.join(os.path.dirname(__file__), "../../frontend/dist")
for _dir in [_dist, _static]:
    if os.path.exists(_dir):
        app.mount("/app", StaticFiles(directory=_dir, html=True), name="static")
        break


@app.get("/")
def root():
    return {"message": "EnNeuro Web API. Frontend: /app  |  API Docs: /docs"}


if __name__ == "__main__":
    uvicorn.run("web_server.main:app", host="0.0.0.0", port=8000, reload=True)
