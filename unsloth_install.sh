#!/bin/bash

echo " =========================================================================== "

conda create -n unsloth_env python=3.11
conda activate unsloth_env

python -m pip install --upgrade pip

echo "!!!! torch --------------------------------------"
#pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install wheel packaging
echo "!!!! flash --------------------------------------"
pip install flash-attn==2.6.3 --no-build-isolation
echo "!!!! unsloth --------------------------------------"
python -m pip install "unsloth[huggingface] @ git+https://github.com/unslothai/unsloth.git"
echo "!!!! bitsandbytes --------------------------------------"
pip install bitsandbytes
echo "!!!! xformers --------------------------------------"
pip install xformers torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
echo "!!!! triton --------------------------------------"
pip install triton==2.1.0 # WTF
