from datetime import datetime
from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

fake_db = {}


class Item(BaseModel):
    title: str
    timestamp: datetime
    description: Union[str, None] = None


app = FastAPI()


@app.put("/items/{id}")
def update_item(id: str, item: Item):
    # 接收一个对象（如 Pydantic 模型），并返回一个 JSON 兼容的版本
    json_compatible_item_data = jsonable_encoder(item)
    fake_db[id] = json_compatible_item_data