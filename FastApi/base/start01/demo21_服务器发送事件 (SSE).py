import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncIterable, Optional
import asyncio

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None


items = [
    Item(name="Plumbus", description="A multi-purpose household device."),
    Item(name="Portal Gun", description="A portal opening device."),
]

# ✅ SSE (Server-Sent Events) 格式
async def generate_sse():
    """生成 SSE 格式的数据流"""
    for item in items:
        # SSE 格式：data: {json}\n\n
        data = json.dumps(item.model_dump(), ensure_ascii=False)
        yield f"data: {data}\n\n"
        await asyncio.sleep(1)  # 模拟延迟

    # 发送结束标记（可选）
    yield "data: {\"status\": \"complete\"}\n\n"


@app.get("/items/sse")
async def stream_items_sse():
    """使用 SSE 流式发送数据"""
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",  # SSE 标准 MIME 类型
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
        }
    )
