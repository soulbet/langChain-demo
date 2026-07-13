from fastapi import FastAPI

description = """
ChimichangApp API helps you do awesome stuff. 🚀

## Items

You can **read items**.

## Users

You will be able to:

* **Create users** (_not implemented_).
* **Read users** (_not implemented_).
"""

app = FastAPI(
    title="ChimichangApp", # 标题
    description=description, # 描述
    summary="Deadpool's favorite app. Nuff said.", # 描述
    version="0.0.1", # 版本
    terms_of_service="http://example.com/terms/",  # 服务条款页面地址
    contact={
        "name": "Deadpoolio the Amazing",
        "url": "http://x-force.example.com/contact/",
        "email": "dp@x-force.example.com",
    }, # 联系信息：包含联系人姓名、网站和邮箱
    license_info={
        "name": "Apache 2.0",
        "url": "https://apache.ac.cn/licenses/LICENSE-2.0.html",
    },  # 许可证信息：使用 Apache 2.0 开源协议
)


@app.get("/items/")
async def read_items():
    return [{"name": "Katana"}]