from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

"""
源是协议 (http, https)、域名 (myapp.com, localhost, localhost.tiangolo.com) 和端口 (80, 443, 8080) 的组合。
"""

"""
FastAPI应用示例 - CORS中间件配置
演示如何配置跨域资源共享(CORS)中间件以允许特定的外部源访问API
"""
app = FastAPI()

# 配置允许跨域访问的源列表
# 注意：列表中包含了通配符模式的域名和端口配置
origins = [
    "https://.tiangolo.com",
    "https://.tiangolo.com",
    "https://",
    "https://:8080",
]

# 添加CORS中间件到应用中
# 配置参数：
# - allow_origins: 允许访问的外部源列表
# - allow_credentials: 是否允许携带认证信息(允许带 cookie / token)
# - allow_methods: 允许的HTTP方法列表，"*"表示允许所有方法
# - allow_headers: 允许的HTTP头列表，"*"表示允许所有头
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main():
    """
    根路径处理函数

    处理GET请求到根路径("/")，返回欢迎消息

    Returns:
        dict: 包含message键的字典，值为欢迎消息字符串
              返回格式: {"message": "Hello World"}
    """
    return {"message": "Hello World"}
