#!/bin/bash
set -e

pip install numpy pillow matplotlib trimesh
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install fvcore iopath

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
export TORCH_CUDA_ARCH_LIST="8.9"
export FORCE_CUDA=1
python setup.py install
cd ..

python -c "
import torch, clip
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRenderer
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('All imports OK')
"