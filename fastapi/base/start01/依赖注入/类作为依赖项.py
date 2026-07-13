from typing import Annotated, Optional

from fastapi import Depends, FastAPI



"""
在 FastAPI 中将一个“可调用对象”作为依赖项传入，它将分析该“可调用对象”的参数，
并以与路径操作函数参数相同的方式处理它们。包括子依赖项
"""

app = FastAPI()


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


class CommonQueryParams:
    def __init__(self, q: Optional[str] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit


@app.get("/items/")
async def read_items(commons: Annotated[CommonQueryParams, Depends(CommonQueryParams)]):
# async def read_items(commons: Annotated[CommonQueryParams, Depends()]): # 上面的简写形式
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})
    return response