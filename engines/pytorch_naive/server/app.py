"""FastAPI app exposing the same /generate contract as the existing
client adapter (client/protocols/pytorch_hf.py).

Routes:

  POST /generate    body: {"prompt": str, "max_tokens": int, "stream": bool}
  GET  /healthz     liveness  — uvicorn is up
  GET  /readyz      readiness — model is loaded
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Config
from .engine import Engine
from .streaming import sse_response


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("pytorch_naive")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(gt=0, le=8192)
    stream: bool = False


cfg = Config.from_env()
engine = Engine(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "loading %s on %s dtype=%s max_model_len=%d",
        cfg.model_name,
        cfg.device,
        cfg.dtype,
        cfg.max_model_len,
    )
    await asyncio.to_thread(engine.load)
    log.info("model ready")
    yield


app = FastAPI(lifespan=lifespan, title="pytorch_naive (E3)")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if not engine.ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready", "model": cfg.model_name}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="model not ready")
    if req.stream:
        return sse_response(engine.generate_stream(req.prompt, req.max_tokens))
    text, prompt_tokens, completion_tokens = await engine.generate_nonstream(
        req.prompt, req.max_tokens
    )
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
