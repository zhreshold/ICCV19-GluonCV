## Setup instructions for your own SageMaker Notebook instances

The following instructions can help you set up your own SageMaker notebooks from the tutorial.

#### In Amazon SageMaker console, create a lifetime configuration named `d2lnlp`

For `Create notebook` script, enter the following

```bash
#!/bin/bash
sudo -u ec2-user -i <<'EOF'

cd ~/SageMaker
wget https://github.com/eric-haibin-lin/AMLC19-GluonNLP/archive/master.zip
unzip master.zip
rm master.zip
mv AMLC19-GluonNLP-master AMLC19-GluonNLP

source activate mxnet_p36
pip install spacy regex
pip install NLTK==3.2.5
pip install -U mxnet-cu100mkl==1.5.0b20190630
pip install -U d2l==0.10
pip install gluonnlp

EOF
```

#### Create notebook instance with Lifetime Configuration `d2lnlp`

When you create your own notebook instance, click the `Additional configuration` and specify `d2lnlp` in `Lifetime configuration - optional` section.
Launch the notebook instance as usual.
