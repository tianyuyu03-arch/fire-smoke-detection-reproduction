import os
import torch
from ultralytics.utils.torch_utils import select_device

print('CUDA_VISIBLE_DEVICES before select_device:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('selected:', select_device(''))
print('CUDA_VISIBLE_DEVICES after select_device:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('logical UUID:', torch.cuda.get_device_properties(0).uuid)
