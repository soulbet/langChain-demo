from typing import Annotated, Union
from fastapi import Depends, FastAPI

"""

用于解耦代码、复用逻辑、管理资源（如数据库连接、认证）

依赖项 (Dependency)：可复用的函数 / 类，提供公共逻辑（认证、数据库、参数校验）。
Depends：声明依赖的装饰器 / 函数，告诉 FastAPI 自动注入。

流程：
定义一个依赖函数
接口写上 = Depends(函数名)
请求进来时，FastAPI 先调用这个依赖函数
拿到函数的 返回值
把这个值注入到接口参数里
"""


app = FastAPI()

# 定义依赖函数
## 参数可以由接口访问参数自动传入
async def common_parameters(q: Union[str, None] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

# 在路由中使用依赖
@app.get("/items/")
async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
    return commons

@app.get("/users/")
async def read_users(commons: Annotated[dict, Depends(common_parameters)]):
    return commons
