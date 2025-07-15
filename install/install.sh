set -e

conda install -c menpo osmesa
conda install -c conda-forge patchelf
conda install -c conda-forge vulkan-tools
conda install -c conda-forge hdfs3

pip install pip==24.0

# RL package
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install stable-baselines3==1.8.0 wandb tensorboard scipy "cython<3" protobuf==4.25 colorama

# set up MetaWorld environment
cd Metaworld
pip install -e .
cd ..

# set up ManiSkill2 environment
cd ManiSkill2
pip install -e .
cd ..

# set up data
cd run_maniskill
bash download_data.sh
cd ..

# set up code generation
pip install transformers langchain_community json5 openai httpx[socks]
pip install langchain chromadb==0.4.0
pip install pandas==1.5.3 matplotlib==3.4 kiwisolver==1.0.1
