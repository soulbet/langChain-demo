from fastapi import FastAPI, Response, Cookie

app = FastAPI()

"""
response.set_cookie(
    key="session_id",
    value="abc123",
    max_age=3600,          # 有效期 1小时（秒）
    expires=3600,          # 过期时间戳
    path="/",              # 哪些路径能访问
    domain=".xxx.com",     # 子域名生效
    secure=True,           # HTTPS 才发送
    httponly=True,         # 禁止 JS 读取，防 XSS
    samesite="lax"         # 防 CSRF
)
"""


# 模拟登录，设置 Cookie
@app.post("/login")
async def login(response: Response, username: str, password: str):
    if username == "admin" and password == "123456":
        # 登录成功，设置 Cookie
        response.set_cookie(
            key="user",
            value=username,
            httponly=True,
            max_age=3600
        )
        return {"msg": "登录成功"}
    return {"msg": "账号或密码错误"}

# 读取 Cookie 验证登录
@app.get("/user/info")
async def user_info(user: str = Cookie(None)):
    if not user:
        return {"msg": "未登录"}
    return {"user": user, "msg": "已登录"}