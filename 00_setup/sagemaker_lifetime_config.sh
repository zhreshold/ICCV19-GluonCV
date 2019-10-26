#!/bin/bash
sudo -u ec2-user -i <<'EOF'

sudo yum install -y llvm6.0-devel
sudo ln -s /usr/lib64/llvm6.0/lib/libLLVM-6.0.so /usr/local/lib/

source /home/ec2-user/anaconda3/bin/activate mxnet_p36

pip install --upgrade mxnet-cu100mkl
pip install gluoncv==0.6.0b20191012
pip install xgboost RISE mmcv opencv-python
# beta prebuilt tvm for sagemaker only
pip install https://haichen-tvm.s3-us-west-2.amazonaws.com/tvm_cu100-0.6.dev0-cp36-cp36m-linux_x86_64.whl
pip install https://haichen-tvm.s3-us-west-2.amazonaws.com/topi-0.6.dev0-py3-none-any.whl
# rise for presentation
jupyter-nbextension install rise --py --user
jupyter nbextension enable rise --user --py

source /home/ec2-user/anaconda3/bin/deactivate

echo "export TVM_NUM_THREADS=4" >> $HOME/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib64/llvm6.0/lib:\$LD_LIBRARY_PATH" >> $HOME/.bashrc


mkdir -p ~/.mxnet
mkdir -p ~/.mxnet/datasets
# Due to 5min limit of SageMaker lifetime config script, we use detached screen to download datasets
# Skip if VOC and COCO datasets are not required

echo "#!/bin/bash" > $HOME/get_data.sh
echo "wget -c --retry-connrefused --tries=10 -P ~/SageMaker  https://zhiz-cache.s3.amazonaws.com/voc_coco.zip" >> $HOME/get_data.sh
echo "cd ~/SageMaker/ && unzip voc_coco.zip" >> $HOME/get_data.sh
echo "ln -s ~/SageMaker/datasets/VOCdevkit ~/.mxnet/datasets/voc" >> $HOME/get_data.sh
echo "ln -s ~/SageMaker/datasets/mscoco ~/.mxnet/datasets/coco" >> $HOME/get_data.sh

cd $HOME
screen -dmS data bash -c 'sleep 1; bash $HOME/get_data.sh'

cd ~/SageMaker/
git clone https://github.com/zhreshold/ICCV19-GluonCV
cd ICCV19-GluonCV
rm -r _static Makefile

EOF
