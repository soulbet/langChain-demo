import os

THRESHOLD = 10 * 1024 * 1024  # 100MB
ignore_file = "../.gitignore"

with open(ignore_file, "a") as f:
    for root, _, files in os.walk("."):
        for file in files:
            path = os.path.join(root, file)
            print(path)
            if os.path.getsize(path) > THRESHOLD:
                f.write(f"{path}\n")
