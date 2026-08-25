import os
from collections import defaultdict
from pathlib import Path

THRESHOLD = 10 * 1024 * 1024  # 100MB
ignore_file = Path(__file__).parent.parent / ".gitignore"
print(ignore_file)

d = {}
with open(ignore_file, "a") as f:
    for root, _, files in os.walk(".."):
        for file in files:
            path = os.path.join(root, file)
            d[path]=os.path.getsize(path)
            if os.path.getsize(path) > THRESHOLD:
                f.write(f"{path}\n")
sd=dict(sorted(d.items(),key=lambda item: item[1], reverse=True))
for i,l in sd.items():
    print(f"{i}: {l}")