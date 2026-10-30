import os
import torch

print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('CUDA device count:', torch.cuda.device_count())
props = torch.cuda.get_device_properties(0)
print('Logical device:', props.name)
print('Logical UUID:', props.uuid)
