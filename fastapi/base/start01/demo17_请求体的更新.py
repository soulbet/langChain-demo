from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    """
    商品数据模型，所有字段均为可选以支持部分更新

    Attributes:
        name: 商品名称，可选
        description: 商品描述，可选
        price: 商品价格，可选
        tax: 税率，默认值为10.5
        tags: 标签列表，默认为空列表
    """
    name: Union[str, None] = None
    description: Union[str, None] = None
    price: Union[float, None] = None
    tax: float = 10.5
    tags: list[str] = []


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}

### 全量更新
@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: str):
    return items[item_id]


@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: str, item: Item):
    update_item_encoded = jsonable_encoder(item)
    items[item_id] = update_item_encoded
    return update_item_encoded



## 部分更新
@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: str):
    """
    根据商品ID获取商品信息

    Args:
        item_id: 商品的唯一标识符

    Returns:
        Item: 对应的商品数据模型实例
    """
    return items[item_id]


@app.patch("/items/{item_id}")
async def update_item(item_id: str, item: Item) -> Item:
    """
    部分更新指定商品的信息

    使用PATCH方法实现部分更新，只更新请求中提供的字段，
    未提供的字段保持原值不变。通过exclude_unset=True识别
    客户端实际提交的字段。

    Args:
        item_id: 要更新的商品的唯一标识符
        item: 包含更新数据的Item模型实例，只有显式设置的字段会被应用

    Returns:
        Item: 更新后的完整商品数据模型实例
    """
    # 获取存储的原始商品数据并转换为Pydantic模型
    stored_item_data = items[item_id]
    stored_item_model = Item(**stored_item_data)

    # 提取请求中显式设置的字段（排除未设置的字段）
    update_data = item.model_dump(exclude_unset=True)

    # 基于原模型创建副本并应用更新数据
    updated_item = stored_item_model.model_copy(update=update_data)

    # 将更新后的模型转换为基础数据类型并存回存储
    items[item_id] = jsonable_encoder(updated_item)
    return updated_item
