## Installation

### ----- Option 1 - Vast.ai -----

Tested on: RTX 3090 (sm_86) / RTX 4090 (sm_89), CUDA 12.x, Python 3.12

### 1. Clone repository
git clone https://github.com/Naes0412/Third-Year-Project-Test-1.git
cd Third-Year-Project-Test-1

### 2. Use a vast.ai instance
When renting on vast.ai:
- Filter by Max CUDA: 12.x (avoid 13.0+)
- Use the **PyTorch (Vast)** template

### 3. Downgrade GCC
apt-get install -y gcc-12 g++-12
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

### 4. Run the setup script
bash setup.sh

> **Note:** `setup.sh` uses `TORCH_CUDA_ARCH_LIST="8.6"` for RTX 3090.
> Change to `8.9` if using an RTX 4090.

### 5. Run
python main_gpu.py

### ----- Option 2 - Google Colab -----

TBA
