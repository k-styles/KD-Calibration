conda create --name kd python=3.8 -y
source activate kd

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch -y

pip install tqdm
pip install transformers
pip install scipy
pip install pandas

git clone https://github.com/bicycleman15/accelerate
cd accelerate
git checkout personal
pip install .
cd ..
rm -rf accelerate

pip install simple-gpu-scheduler