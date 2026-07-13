from typing import Annotated

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# 将路径操作函数参数的类型声明为 bytes 所有内容都将存储在内存中。这对于小文件非常适用
@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

# 文件存储在内存中，直到达到最大限制，超过该限制后将存储在磁盘上
## 合处理大型文件，如图像、视频、大型二进制文件等
### 可以获取上传文件的元数据
#### 具有类文件 (file-like) 的 async 接口
"""
UploadFile 具有以下属性：

filename：包含上传文件原始名称的 str（例如 myimage.jpg）。
content_type：包含内容类型（MIME 类型 / 媒体类型）的 str（例如 image/jpeg）。
file：一个 SpooledTemporaryFile（一个类文件对象）。这是实际的 Python 文件对象，你可以将其直接传递给其他需要“类文件”对象的函数或库。


UploadFile 具有以下 async 方法。它们在底层（使用内部的 SpooledTemporaryFile）调用相应的文件方法。

write(data)：将 data（str 或 bytes）写入文件。
read(size)：读取文件的 size（int）字节/字符。
seek(offset)：跳转到文件中的字节位置 offset（int）。
例如，await myfile.seek(0) 将跳转到文件开头。
如果你已经运行过 await myfile.read() 但还需要再次读取内容，这特别有用。
close()：关闭文件。
"""
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}