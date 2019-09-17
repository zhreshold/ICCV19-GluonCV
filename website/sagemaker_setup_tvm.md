## Setup instructions for your own SageMaker Notebook instances for TVM

The following instructions can help you set up your own SageMaker notebooks from the tutorial.

#### In Amazon SageMaker console, create a lifetime configuration named `d2lnlptvm`

For `Create notebook` script, enter the following

```bash
#!/bin/bash
sudo -u ec2-user -i <<'EOF'

sudo yum install -y llvm6.0-devel
source /home/ec2-user/anaconda3/bin/activate mxnet_p36

pip install --upgrade mxnet-mkl==1.5.0b20190630
pip install gluonnlp
pip install https://haichen-tvm.s3-us-west-2.amazonaws.com/tvm-0.6.dev0-cp36-cp36m-linux_x86_64.whl
pip install https://haichen-tvm.s3-us-west-2.amazonaws.com/topi-0.6.dev0-py3-none-any.whl

source /home/ec2-user/anaconda3/bin/deactivate

echo "export TVM_NUM_THREADS=18" >> $HOME/.bashrc

cd ~/SageMaker && git clone https://github.com/eric-haibin-lin/AMLC19-GluonNLP.git

EOF
```

#### Create notebook instance with Lifetime Configuration `d2lnlptvm`

When you create your own notebook instance, click the `Additional configuration` and specify `d2lnlptvm` in `Lifetime configuration - optional` section.
Launch the notebook instance as usual.
