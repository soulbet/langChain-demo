import requests

url = "http://172.31.148.43:55004/embedding"

r = requests.post(
    url,
    json={"input": "你好"}
)

print(r.status_code)
print(r.text)