import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncIterable, Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None

items = [
    Item(name="Plumbus", description="A multi-purpose household device."),
    Item(name="Portal Gun", description="A portal opening device."),
]

# ✅ 这才是 JSON Lines
async def generate_jsonl():
    for item in items:
        yield json.dumps(item.model_dump(), ensure_ascii=False) + "\n"

@app.get("/items/stream")
async def stream_items():
    return StreamingResponse(
        generate_jsonl(),
        media_type="application/jsonlines"  # 标准类型
    )