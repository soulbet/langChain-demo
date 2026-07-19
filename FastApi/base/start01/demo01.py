import uvicorn
from fastapi import FastAPI

app = FastAPI()

"""
启动命令：fastapi dev 文件路径
方位地址：http://127.0.0.1:8000/items/5?q=somequery
文档：http://127.0.0.1:8000/docs#/default/read_root__get
"""

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q:str = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)