from fastapi import Depends, FastAPI

from .dependencies import get_query_token, get_token_header
from .internal import admin

from .routers import items,users

app = FastAPI(dependencies=[Depends(get_query_token)])

"""
把不同模块的接口，写在不同的文件里，最后统一挂载到 FastAPI 上。
"""

app.include_router(users.router)
app.include_router(items.router)
app.include_router(
    admin.router,
    prefix="/admin", # 添加前缀
    tags=["admin"], # 添加标签
    dependencies=[Depends(get_token_header)], # 添加依赖
    responses={418: {"description": "I'm a teapot"}}, # 添加响应
)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}