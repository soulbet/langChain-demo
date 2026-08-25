import requests

url = "http://172.31.148.43:55004/embedding"

for n in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200,300,400,500,800,1000]:
    text = "测试文本" * n

    try:
        r = requests.post(
            url,
            json={"input": text},
            timeout=60
        )

        print(
            f"重复次数={n:3d} | "
            f"字符数={len(text):4d} | "
            f"状态码={r.status_code}"
        )

    except Exception as e:
        print(f"重复次数={n}: ERROR {e}")