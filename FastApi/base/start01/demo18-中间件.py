import time

from fastapi import FastAPI, Request

"""
中间件：每个请求在被任何特定的路径操作前对其进行处理 在每个 响应 返回之前对其进行处理
"""

app = FastAPI()


"""
request（请求对象）。
一个 call_next 函数，它将接收 request 作为参数。
此函数会将 request 传递给相应的 路径操作。
然后它返回由相应的 路径操作 生成的 response（响应对象）。
之后，你可以在返回 response 之前对其进行进一步修改。

"""

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response