
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

## 把电脑上的 static 文件夹，变成一个可以直接通过网址访问的 “公开文件夹”！

app.mount("/static", StaticFiles(directory="static"), name="static")