import torch
import torchaudio
import torchvision

print(torch.__version__)  # 應該顯示 2.4.1+cu118
print(torchaudio.__version__)  # 應該顯示 2.4.1+cu118
print(torchvision.__version__)  # 應該顯示 0.19.1+cu118

print(torch.cuda.is_available())
print(torch.backends.cudnn.version())

import sys
sample_memory = (torch.randn(1, 10, 50), torch.randn(1, 2, 50), 0, 0.0, torch.randn(1, 1, 10, 50), torch.randn(1, 2, 50), False)
memory_size = 150000
print(f"Size of one memory entry: {sys.getsizeof(sample_memory)} bytes")
print(f"Total memory for 150,000 entries: {sys.getsizeof(sample_memory) * memory_size / 1024**2:.2f} MB")

