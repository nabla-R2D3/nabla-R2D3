PROJECT_DIR=$(pwd)
conda create -n NablaR2D3 python=3.9
conda activate NablaR2D3
# Pytorch
pip3 install -i https://download.pytorch.org/whl/cu121 -U torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip3 install -i https://download.pytorch.org/whl/cu121 -U xformers==0.0.27

# A modified gaussian splatting (+ alpha, depth, normal rendering)
cd extensions && git clone https://github.com/BaowenZ/RaDe-GS.git --recursive && cd RaDe-GS/submodules
pip3 install ./diff-gaussian-rasterization

cd ${PROJECT_DIR}
cd extensions
pip3 install -e git+https://github.com/tgxs002/HPSv2.git@866735ecaae999fa714bd9edfa05aa2672669ee3#egg=hpsv2
cd ${PROJECT_DIR}
# Others
cd extensions/normal_priors/StableNormal
pip3 install -e ./
cd ${PROJECT_DIR}
# Others
pip3 install -U gpustat
pip3 install -U -r settings/requirements.txt
sudo apt-get install -y ffmpeg
