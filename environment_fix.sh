

conda init pso
conda activate pso

conda config --set channel_priority strict

conda install -y --override-channels -c pytorch -c nvidia -c defaults \
  pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 \
  typing_extensions libpng

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("built with CUDA:", torch.backends.cuda.is_built())
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

pip install "diffusers==0.27.2" "huggingface_hub<0.23" -U

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

pip install -U charset-normalizer

