import random
from typing import Annotated, Union

from fastapi import FastAPI, Query
from pydantic import AfterValidator

app = FastAPI()

data = {
    "isbn-9781529046137": "The Hitchhiker's Guide to the Galaxy",
    "imdb-tt0371724": "The Hitchhiker's Guide to the Galaxy",
    "isbn-9781439512982": "Isaac Asimov: The Complete Stories, Vol. 2",
}


def check_valid_id(id: str):
    if not id.startswith(("isbn-", "imdb-")):
        raise ValueError('Invalid ID format, it must start with "isbn-" or "imdb-"')
    return id

@app.get("/items/")
async def read_items(id: Annotated[Union[str, None], AfterValidator(check_valid_id)] = None, # 自定义校验
    q: Annotated[   ## 参数添加元数据
        Union[str, None],
        Query(
            alias="item-query",  ## 别名，不符合python命名时使用
            title="Query string",  ##
            description="Query string for the items to search in the database that have a good match",  ## 说明
            min_length=3,  ## 接收的参数最小长度
            max_length=50,  ## 接收的参数最大长度
            pattern="^fixedquery$", ## 正则匹配接收的参数
            deprecated=True,  ## 表示已弃用
        ),
    ] = ["foo", "bar"], ## 默认值
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}

    if id:
        item = data.get(id)
    else:
        id, item = random.choice(list(data.items()))
    if q:
        results.update({"q": q})
    return results