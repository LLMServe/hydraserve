# Run this script as root on master and every worker node

if [ $# -ne 1 ];
then
    echo "Enter 0 for master and 1 for worker"
    exit
fi

yum install -y git
wget --user-agent="Mozilla" https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.1-Linux-x86_64.sh
sh Anaconda3-5.3.1-Linux-x86_64.sh -b
echo ". /root/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source /root/anaconda3/etc/profile.d/conda.sh
cp condarc ~/.condarc
cp -r dist ~
conda clean -i

cd ~
git clone https://github.com/gjghfd/ServerlessLLM
cd ServerlessLLM
if [ $1 == "0" ];
then
    conda create -n sllm python=3.10 -y
    conda activate sllm
    pip install -e .
    pip install modelscope
else
    git clone https://github.com/gjghfd/ServerlessLLM
    cd ServerlessLLM
    conda create -n sllm-worker python=3.10 -y
    conda activate sllm-worker
    pip install -e ".[worker]"
    pip install ~/dist/*.whl
fi