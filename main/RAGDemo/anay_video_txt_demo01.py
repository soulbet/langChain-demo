from langchain_community.document_loaders import BiliBiliLoader
# pip install bilibili-api-python yt-dlp openai-whisper
# 填你自己的cookie
SESSDATA = "93881f93%2C1793004510%2C16590%2A41CjCoAUiMs-UHjKcw5YWyBLw0ErMnswo7e6EE2u_m_U9tHEAB0xDquO7EP7eBTZoZdxcSVlhtdS1vOVA0RjliZXNoTUotMkVJVWJLRG85Wm1ldE5YTkVHbmtOc1VyVnh4cmFSeFJfRE00ekFLVTZrUUdtckZxR2NxSmtmMkNROGRNRmhiWVBsVmR3IIEC"
BILI_JCT = "f49ce679e4de9ca041cf7e42f0934ba9"
BUVID3 = "689C74C7-4237-99D9-0666-9CCA0498647587826infoc"

# 视频URL列表
urls = [
    "https://www.bilibili.com/video/BV1oroDB1EjV/?spm_id_from=333.1007.tianma.1-1-1.click&vd_source=b2f3779d69544250cabdb9e8f901e9a0"
]

# 万能加载 B 站视频（自动语音转文字，不怕没字幕）
def load_bilibili_video(url: str):
    # 1. 下载音频
    audio_file = "bilibili_audio.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
        "outtmpl": audio_file,
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # 2. 语音转文字（支持中文）
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, language="zh", fp16=False)

    # 3. 生成 LangChain Document
    return Document(
        page_content=result["text"],
        metadata={"source": url, "type": "bilibili"}
    )

loader = BiliBiliLoader(
    video_urls=urls,
    sessdata=SESSDATA,
    bili_jct=BILI_JCT,
    buvid3=BUVID3
)

docs = loader.load()

# 输出结果
for doc in docs:
    print("=== 视频内容 ===")
    print(doc.page_content)
    print("=== 元数据 ===")
    print(doc.metadata)