import torch
import torchvision

print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
