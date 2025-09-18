import torch
import torchaudio
import torchvision

print(torch.__version__)  # 應該顯示 2.4.1+cu118
print(torchaudio.__version__)  # 應該顯示 2.4.1+cu118
print(torchvision.__version__)  # 應該顯示 0.19.1+cu118

print(torch.cuda.is_available())
print(torch.backends.cudnn.version())

