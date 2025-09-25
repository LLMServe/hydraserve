yum install -y python38
cd /usr/bin
rm -f python
rm -f pip
ln -s python3.8 python
ln -s pip3.8 pip
pip install kubernetes==31.0.0 modelscope==1.15.0 requests openai fastapi aiohttp uvicorn[standard] matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
python -V
pip -V