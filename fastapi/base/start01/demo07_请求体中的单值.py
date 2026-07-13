from typing import Annotated

from fastapi import Body, FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float= None


class User(BaseModel):
    username: str
    full_name: str = None


@app.put("/items/{item_id}")
async def update_item(
    item_id: int, item: Annotated[Item, Body(embed=True)], user: User, importance: Annotated[int, Body()]
):
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    return results