from typing import Annotated

from fastapi import FastAPI, Form
from pydantic import BaseModel

app = FastAPI()


class FormData(BaseModel):
    username: str
    password: str
    model_config = {"extra": "forbid"} # 限制表单字段只能包含 Pydantic 模型中声明的字段，并禁止任何额外的字段


@app.post("/login/")
async def login(data: Annotated[FormData, Form()]):
    return data