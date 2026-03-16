import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)