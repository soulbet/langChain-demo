import torch
import numpy
import sklearn
import transformers

print("numpy版本：", numpy.__version__)
print("sklearn版本：", sklearn.__version__)
print("torch版本：", torch.__version__)
# 核心校验GPU
print("CUDA是否可用：", torch.cuda.is_available())
print("GPU名称：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU")
print("transformers版本：", transformers.__version__)