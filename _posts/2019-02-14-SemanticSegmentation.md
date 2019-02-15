---
layout: post
title: Segmentation
---

## Configuration

First we install the necessary python libraries


```python
# !pip install gluoncv --pre
# !pip install mxnet-cu90 --pre
```

We also need to download the Pascal VOC dataset for our training demonstration later.


```python
# !python pascal_voc.py
```


```python
import random
import matplotlib.image as mpimg
from datetime import datetime

import numpy as np
import mxnet as mx
from mxnet import image, autograd
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.utils import viz
from gluoncv.loss import MixSoftmaxCrossEntropyLoss
from gluoncv.utils.parallel import *

ctx = mx.gpu(0)
```

## Segmentation with pre-trained models

We first download an image for the demo


```python
url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/1.jpg'
filename = 'example.jpg'
gluoncv.utils.download(url, filename)
input_img = image.imread(filename)

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (15, 9)

viz.plot_image(input_img)
plt.show()
```


    <Figure size 1500x900 with 1 Axes>


Next we define the transformation functions. It is just transformation and normalization.


```python
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
img = transform_fn(input_img)
img = img.expand_dims(0).as_in_context(ctx)
img.shape
```




    (1, 3, 400, 500)



We'll use the pre-trained model `fcn_resnet101_voc`.


```python
net = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True, ctx=ctx)
```

Note we call `net.demo` here to make prediction.


```python
output = net.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
```

With the prediction, we can extract the mask and check how it works.


```python
mask = viz.get_color_pallete(predict, 'pascal_voc')
mask = np.array(mask.convert('RGB'), dtype=np.int)

combined = (mask+input_img.asnumpy())/2
plt.imshow(combined.astype(np.uint8))
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_14_0.png)


Next, let's try a more difficult image.


```python
filename = 'streetview_amazon.png'
input_img = image.imread(filename)
img = transform_fn(input_img)
img = img.expand_dims(0).as_in_context(ctx)

viz.plot_image(input_img)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_16_0.png)


The image is larger thus takes a bit more time to predict.


```python
output = net.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

mask = viz.get_color_pallete(predict, 'pascal_voc')
mask = np.array(mask.convert('RGB'), dtype=np.int)

combined = (mask+input_img.asnumpy())/2
plt.imshow(combined.astype(np.uint8))
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_18_0.png)


We merely see anything. This is because the dataset Pascal VOC doesn't have too much labels for the objects in the above image.

Instead, we use another model pre-trained on the dataset ADE20k.


```python
net = gluoncv.model_zoo.get_model('fcn_resnet50_ade', pretrained=True, ctx=ctx)
output = net.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

mask = viz.get_color_pallete(predict, 'ade20k')
mask = np.array(mask.convert('RGB'), dtype=np.int)

combined = (mask+input_img.asnumpy())/2
plt.imshow(combined.astype(np.uint8))
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_20_0.png)


How about an even more difficult one?


```python
filename = 'streetview_meguro.jpg'
input_img = image.imread(filename)
img = transform_fn(input_img)
img = img.expand_dims(0).as_in_context(ctx)

viz.plot_image(input_img)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_22_0.png)



```python
output = net.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

mask = viz.get_color_pallete(predict, 'ade20k')
mask = np.array(mask.convert('RGB'), dtype=np.int)

combined = (mask+input_img.asnumpy())/2
plt.imshow(combined.astype(np.uint8))
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_23_0.png)


Our model `fcn_resnet50_ade` overfits the sky, and is confused by the plant on the wall.

Let's try a larger one: `deeplab_resnet101_ade`.


```python
net = gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True, ctx=ctx)
output = net.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

mask = viz.get_color_pallete(predict, 'ade20k')
mask = np.array(mask.convert('RGB'), dtype=np.int)

combined = (mask+input_img.asnumpy())/2
plt.imshow(combined.astype(np.uint8))
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_25_0.png)


It is much cleaner. Although it is still a bit confused by the plant on the wall.

Remember, choose your pre-trained model based on the data.

## Training on VOC

Let's start the training of a small network `fcn_resnet50_voc`. This time we don't start from a pre-trained model, but instead we start from a randomly initialized one.

You may increase the training process to see how the quality improved.

First, let's load the model without pretrained weight.


```python
net = gluoncv.model_zoo.get_model('fcn_resnet50_voc', pretrained=False)
```

Come back to the first image, how does the output looks like?


```python
input_img = image.imread('example.jpg')
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
img = transform_fn(input_img)
img = img.expand_dims(0)

output = net.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
mask = viz.get_color_pallete(predict, 'pascal_voc')
mask = np.array(mask.convert('RGB'), dtype=np.int)

combined = (mask+input_img.asnumpy())/2
plt.imshow(combined.astype(np.uint8))
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_30_0.png)


It is just random noise.

Notice: after called `net.demo()`, we need to re-define the size of `net`'s output. 

Because by calling `net.demo()` we changed the size to match the input image's, while during training `net` need a 480x480 output size.


```python
from mxnet.gluon.data.vision import transforms
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])
net._up_kwargs['height'] = 480
net._up_kwargs['width'] = 480
```

Next we prepare the training data.


```python
trainset = gluoncv.data.VOCSegmentation(split='train', transform=input_transform)
print('Training images:', len(trainset))
# set batch_size = 2 for toy example
batch_size = 4*4
# Create Training Loader
train_data = mx.gluon.data.DataLoader(
    trainset, batch_size, shuffle=True, last_batch='rollover',
    num_workers=batch_size)
```

    Training images: 2913


We randomly sample an image and its mask. This is the usual format for most segmentation training tasks.


```python
random.seed(datetime.now())
idx = random.randint(0, len(trainset))
img, mask = trainset[idx]
# get color pallete for visualize mask
mask = viz.get_color_pallete(mask.asnumpy(), dataset='pascal_voc')
mask.save('mask.png')
# denormalize the image
img = viz.DeNormalize([.485, .456, .406], [.229, .224, .225])(img)
img = np.transpose((img.asnumpy()*255).astype(np.uint8), (1, 2, 0))

# subplot 1 for img
fig = plt.figure()
fig.add_subplot(1,2,1)

plt.imshow(img)
# subplot 2 for the mask
mmask = mpimg.imread('mask.png')
fig.add_subplot(1,2,2)
plt.imshow(mmask)
# display
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_36_0.png)


Next we define the loss, learning rate schedule.


```python
criterion = MixSoftmaxCrossEntropyLoss(aux=True)
lr_scheduler = gluoncv.utils.LRScheduler(mode='poly', baselr=0.0001, niters=len(train_data),
                                         nepochs=50)
```

We use the `DataParallel` interface for model and loss computation. This is useful when Synchronized BatchNorm training is necessary.


```python
ctx_list = [mx.gpu(i) for i in range(4)]
net = DataParallelModel(net, ctx_list)
criterion = DataParallelCriterion(criterion, ctx_list)
```

Next we define the optimizer.


```python

optimizer = mx.gluon.Trainer(net.module.collect_params(), 'sgd',
                             {'lr_scheduler': lr_scheduler,
                              'wd':0.0001,
                              'momentum': 0.9,
                              'multi_precision': True})
```

Now we can start to train!

Remember that due to time limit we will not train until it converges.

However even by only 5 epochs' training, the result starts to make sense.


```python
for epoch in range(5):
    train_loss = 0.0
    for i, (data, target) in enumerate(train_data):
        lr_scheduler.update(i, epoch)
        with autograd.record(True):
            outputs = net(data)
            losses = criterion(outputs, target)
            mx.nd.waitall()
            autograd.backward(losses)
        optimizer.step(batch_size)
        for loss in losses:
            train_loss += loss.asnumpy()[0] / len(losses)
        print('Epoch %d, batch %d, training loss %.3f'%(epoch, i, train_loss/(i+1)))
```

    Epoch 0, batch 0, training loss 3.903
    Epoch 0, batch 1, training loss 3.977
    Epoch 0, batch 2, training loss 3.932
    Epoch 0, batch 3, training loss 3.929
    Epoch 0, batch 4, training loss 3.906
    Epoch 0, batch 5, training loss 3.893
    Epoch 0, batch 6, training loss 3.852
    Epoch 0, batch 7, training loss 3.848
    Epoch 0, batch 8, training loss 3.837
    Epoch 0, batch 9, training loss 3.821
    Epoch 0, batch 10, training loss 3.787
    Epoch 0, batch 11, training loss 3.770
    Epoch 0, batch 12, training loss 3.752
    Epoch 0, batch 13, training loss 3.708
    Epoch 0, batch 14, training loss 3.645
    Epoch 0, batch 15, training loss 3.627
    Epoch 0, batch 16, training loss 3.604
    Epoch 0, batch 17, training loss 3.584
    Epoch 0, batch 18, training loss 3.558
    Epoch 0, batch 19, training loss 3.537
    Epoch 0, batch 20, training loss 3.530
    Epoch 0, batch 21, training loss 3.507
    Epoch 0, batch 22, training loss 3.476
    Epoch 0, batch 23, training loss 3.450
    Epoch 0, batch 24, training loss 3.443
    Epoch 0, batch 25, training loss 3.420
    Epoch 0, batch 26, training loss 3.389
    Epoch 0, batch 27, training loss 3.366
    Epoch 0, batch 28, training loss 3.340
    Epoch 0, batch 29, training loss 3.324
    Epoch 0, batch 30, training loss 3.309
    Epoch 0, batch 31, training loss 3.286
    Epoch 0, batch 32, training loss 3.255
    Epoch 0, batch 33, training loss 3.232
    Epoch 0, batch 34, training loss 3.191
    Epoch 0, batch 35, training loss 3.190
    Epoch 0, batch 36, training loss 3.164
    Epoch 0, batch 37, training loss 3.148
    Epoch 0, batch 38, training loss 3.130
    Epoch 0, batch 39, training loss 3.105
    Epoch 0, batch 40, training loss 3.095
    Epoch 0, batch 41, training loss 3.063
    Epoch 0, batch 42, training loss 3.042
    Epoch 0, batch 43, training loss 3.028
    Epoch 0, batch 44, training loss 2.999
    Epoch 0, batch 45, training loss 2.982
    Epoch 0, batch 46, training loss 2.966
    Epoch 0, batch 47, training loss 2.947
    Epoch 0, batch 48, training loss 2.941
    Epoch 0, batch 49, training loss 2.916
    Epoch 0, batch 50, training loss 2.888
    Epoch 0, batch 51, training loss 2.897
    Epoch 0, batch 52, training loss 2.886
    Epoch 0, batch 53, training loss 2.863
    Epoch 0, batch 54, training loss 2.876
    Epoch 0, batch 55, training loss 2.849
    Epoch 0, batch 56, training loss 2.832
    Epoch 0, batch 57, training loss 2.827
    Epoch 0, batch 58, training loss 2.820
    Epoch 0, batch 59, training loss 2.809
    Epoch 0, batch 60, training loss 2.790
    Epoch 0, batch 61, training loss 2.771
    Epoch 0, batch 62, training loss 2.755
    Epoch 0, batch 63, training loss 2.743
    Epoch 0, batch 64, training loss 2.739
    Epoch 0, batch 65, training loss 2.722
    Epoch 0, batch 66, training loss 2.709
    Epoch 0, batch 67, training loss 2.704
    Epoch 0, batch 68, training loss 2.683
    Epoch 0, batch 69, training loss 2.676
    Epoch 0, batch 70, training loss 2.661
    Epoch 0, batch 71, training loss 2.642
    Epoch 0, batch 72, training loss 2.631
    Epoch 0, batch 73, training loss 2.622
    Epoch 0, batch 74, training loss 2.616
    Epoch 0, batch 75, training loss 2.615
    Epoch 0, batch 76, training loss 2.607
    Epoch 0, batch 77, training loss 2.595
    Epoch 0, batch 78, training loss 2.603
    Epoch 0, batch 79, training loss 2.598
    Epoch 0, batch 80, training loss 2.591
    Epoch 0, batch 81, training loss 2.571
    Epoch 0, batch 82, training loss 2.575
    Epoch 0, batch 83, training loss 2.563
    Epoch 0, batch 84, training loss 2.555
    Epoch 0, batch 85, training loss 2.545
    Epoch 0, batch 86, training loss 2.531
    Epoch 0, batch 87, training loss 2.515
    Epoch 0, batch 88, training loss 2.518
    Epoch 0, batch 89, training loss 2.503
    Epoch 0, batch 90, training loss 2.499
    Epoch 0, batch 91, training loss 2.486
    Epoch 0, batch 92, training loss 2.471
    Epoch 0, batch 93, training loss 2.478
    Epoch 0, batch 94, training loss 2.475
    Epoch 0, batch 95, training loss 2.476
    Epoch 0, batch 96, training loss 2.474
    Epoch 0, batch 97, training loss 2.465
    Epoch 0, batch 98, training loss 2.454
    Epoch 0, batch 99, training loss 2.446
    Epoch 0, batch 100, training loss 2.440
    Epoch 0, batch 101, training loss 2.427
    Epoch 0, batch 102, training loss 2.413
    Epoch 0, batch 103, training loss 2.411
    Epoch 0, batch 104, training loss 2.400
    Epoch 0, batch 105, training loss 2.397
    Epoch 0, batch 106, training loss 2.394
    Epoch 0, batch 107, training loss 2.393
    Epoch 0, batch 108, training loss 2.388
    Epoch 0, batch 109, training loss 2.381
    Epoch 0, batch 110, training loss 2.369
    Epoch 0, batch 111, training loss 2.370
    Epoch 0, batch 112, training loss 2.360
    Epoch 0, batch 113, training loss 2.357
    Epoch 0, batch 114, training loss 2.347
    Epoch 0, batch 115, training loss 2.346
    Epoch 0, batch 116, training loss 2.346
    Epoch 0, batch 117, training loss 2.341
    Epoch 0, batch 118, training loss 2.330
    Epoch 0, batch 119, training loss 2.322
    Epoch 0, batch 120, training loss 2.315
    Epoch 0, batch 121, training loss 2.309
    Epoch 0, batch 122, training loss 2.309
    Epoch 0, batch 123, training loss 2.318
    Epoch 0, batch 124, training loss 2.312
    Epoch 0, batch 125, training loss 2.306
    Epoch 0, batch 126, training loss 2.309
    Epoch 0, batch 127, training loss 2.299
    Epoch 0, batch 128, training loss 2.294
    Epoch 0, batch 129, training loss 2.286
    Epoch 0, batch 130, training loss 2.278
    Epoch 0, batch 131, training loss 2.268
    Epoch 0, batch 132, training loss 2.269
    Epoch 0, batch 133, training loss 2.269
    Epoch 0, batch 134, training loss 2.269
    Epoch 0, batch 135, training loss 2.264
    Epoch 0, batch 136, training loss 2.256
    Epoch 0, batch 137, training loss 2.261
    Epoch 0, batch 138, training loss 2.262
    Epoch 0, batch 139, training loss 2.264
    Epoch 0, batch 140, training loss 2.263
    Epoch 0, batch 141, training loss 2.262
    Epoch 0, batch 142, training loss 2.260
    Epoch 0, batch 143, training loss 2.256
    Epoch 0, batch 144, training loss 2.254
    Epoch 0, batch 145, training loss 2.254
    Epoch 0, batch 146, training loss 2.247
    Epoch 0, batch 147, training loss 2.245
    Epoch 0, batch 148, training loss 2.244
    Epoch 0, batch 149, training loss 2.241
    Epoch 0, batch 150, training loss 2.236
    Epoch 0, batch 151, training loss 2.229
    Epoch 0, batch 152, training loss 2.225
    Epoch 0, batch 153, training loss 2.219
    Epoch 0, batch 154, training loss 2.216
    Epoch 0, batch 155, training loss 2.215
    Epoch 0, batch 156, training loss 2.213
    Epoch 0, batch 157, training loss 2.207
    Epoch 0, batch 158, training loss 2.202
    Epoch 0, batch 159, training loss 2.196
    Epoch 0, batch 160, training loss 2.203
    Epoch 0, batch 161, training loss 2.196
    Epoch 0, batch 162, training loss 2.196
    Epoch 0, batch 163, training loss 2.193
    Epoch 0, batch 164, training loss 2.191
    Epoch 0, batch 165, training loss 2.187
    Epoch 0, batch 166, training loss 2.179
    Epoch 0, batch 167, training loss 2.177
    Epoch 0, batch 168, training loss 2.172
    Epoch 0, batch 169, training loss 2.169
    Epoch 0, batch 170, training loss 2.167
    Epoch 0, batch 171, training loss 2.163
    Epoch 0, batch 172, training loss 2.161
    Epoch 0, batch 173, training loss 2.161
    Epoch 0, batch 174, training loss 2.163
    Epoch 0, batch 175, training loss 2.164
    Epoch 0, batch 176, training loss 2.169
    Epoch 0, batch 177, training loss 2.167
    Epoch 0, batch 178, training loss 2.167
    Epoch 0, batch 179, training loss 2.163
    Epoch 0, batch 180, training loss 2.162
    Epoch 0, batch 181, training loss 2.158
    Epoch 1, batch 0, training loss 0.980
    Epoch 1, batch 1, training loss 1.417
    Epoch 1, batch 2, training loss 1.278
    Epoch 1, batch 3, training loss 1.418
    Epoch 1, batch 4, training loss 1.484
    Epoch 1, batch 5, training loss 1.590
    Epoch 1, batch 6, training loss 1.629
    Epoch 1, batch 7, training loss 1.605
    Epoch 1, batch 8, training loss 1.549
    Epoch 1, batch 9, training loss 1.643
    Epoch 1, batch 10, training loss 1.646
    Epoch 1, batch 11, training loss 1.629
    Epoch 1, batch 12, training loss 1.625
    Epoch 1, batch 13, training loss 1.592
    Epoch 1, batch 14, training loss 1.610
    Epoch 1, batch 15, training loss 1.596
    Epoch 1, batch 16, training loss 1.618
    Epoch 1, batch 17, training loss 1.613
    Epoch 1, batch 18, training loss 1.640
    Epoch 1, batch 19, training loss 1.677
    Epoch 1, batch 20, training loss 1.642
    Epoch 1, batch 21, training loss 1.678
    Epoch 1, batch 22, training loss 1.663
    Epoch 1, batch 23, training loss 1.670
    Epoch 1, batch 24, training loss 1.656
    Epoch 1, batch 25, training loss 1.618
    Epoch 1, batch 26, training loss 1.605
    Epoch 1, batch 27, training loss 1.592
    Epoch 1, batch 28, training loss 1.633
    Epoch 1, batch 29, training loss 1.624
    Epoch 1, batch 30, training loss 1.624
    Epoch 1, batch 31, training loss 1.619
    Epoch 1, batch 32, training loss 1.607
    Epoch 1, batch 33, training loss 1.589
    Epoch 1, batch 34, training loss 1.592
    Epoch 1, batch 35, training loss 1.595
    Epoch 1, batch 36, training loss 1.627
    Epoch 1, batch 37, training loss 1.622
    Epoch 1, batch 38, training loss 1.622
    Epoch 1, batch 39, training loss 1.645
    Epoch 1, batch 40, training loss 1.624
    Epoch 1, batch 41, training loss 1.630
    Epoch 1, batch 42, training loss 1.631
    Epoch 1, batch 43, training loss 1.640
    Epoch 1, batch 44, training loss 1.635
    Epoch 1, batch 45, training loss 1.640
    Epoch 1, batch 46, training loss 1.640
    Epoch 1, batch 47, training loss 1.658
    Epoch 1, batch 48, training loss 1.647
    Epoch 1, batch 49, training loss 1.656
    Epoch 1, batch 50, training loss 1.647
    Epoch 1, batch 51, training loss 1.661
    Epoch 1, batch 52, training loss 1.663
    Epoch 1, batch 53, training loss 1.658
    Epoch 1, batch 54, training loss 1.672
    Epoch 1, batch 55, training loss 1.685
    Epoch 1, batch 56, training loss 1.686
    Epoch 1, batch 57, training loss 1.700
    Epoch 1, batch 58, training loss 1.694
    Epoch 1, batch 59, training loss 1.693
    Epoch 1, batch 60, training loss 1.696
    Epoch 1, batch 61, training loss 1.687
    Epoch 1, batch 62, training loss 1.670
    Epoch 1, batch 63, training loss 1.678
    Epoch 1, batch 64, training loss 1.686
    Epoch 1, batch 65, training loss 1.684
    Epoch 1, batch 66, training loss 1.683
    Epoch 1, batch 67, training loss 1.704
    Epoch 1, batch 68, training loss 1.706
    Epoch 1, batch 69, training loss 1.719
    Epoch 1, batch 70, training loss 1.723
    Epoch 1, batch 71, training loss 1.721
    Epoch 1, batch 72, training loss 1.728
    Epoch 1, batch 73, training loss 1.719
    Epoch 1, batch 74, training loss 1.732
    Epoch 1, batch 75, training loss 1.738
    Epoch 1, batch 76, training loss 1.730
    Epoch 1, batch 77, training loss 1.724
    Epoch 1, batch 78, training loss 1.732
    Epoch 1, batch 79, training loss 1.727
    Epoch 1, batch 80, training loss 1.720
    Epoch 1, batch 81, training loss 1.721
    Epoch 1, batch 82, training loss 1.736
    Epoch 1, batch 83, training loss 1.732
    Epoch 1, batch 84, training loss 1.726
    Epoch 1, batch 85, training loss 1.725
    Epoch 1, batch 86, training loss 1.719
    Epoch 1, batch 87, training loss 1.725
    Epoch 1, batch 88, training loss 1.722
    Epoch 1, batch 89, training loss 1.715
    Epoch 1, batch 90, training loss 1.711
    Epoch 1, batch 91, training loss 1.705
    Epoch 1, batch 92, training loss 1.709
    Epoch 1, batch 93, training loss 1.703
    Epoch 1, batch 94, training loss 1.708
    Epoch 1, batch 95, training loss 1.711
    Epoch 1, batch 96, training loss 1.706
    Epoch 1, batch 97, training loss 1.699
    Epoch 1, batch 98, training loss 1.699
    Epoch 1, batch 99, training loss 1.701
    Epoch 1, batch 100, training loss 1.704
    Epoch 1, batch 101, training loss 1.707
    Epoch 1, batch 102, training loss 1.703
    Epoch 1, batch 103, training loss 1.709
    Epoch 1, batch 104, training loss 1.699
    Epoch 1, batch 105, training loss 1.695
    Epoch 1, batch 106, training loss 1.696
    Epoch 1, batch 107, training loss 1.706
    Epoch 1, batch 108, training loss 1.706
    Epoch 1, batch 109, training loss 1.710
    Epoch 1, batch 110, training loss 1.716
    Epoch 1, batch 111, training loss 1.712
    Epoch 1, batch 112, training loss 1.712
    Epoch 1, batch 113, training loss 1.714
    Epoch 1, batch 114, training loss 1.711
    Epoch 1, batch 115, training loss 1.720
    Epoch 1, batch 116, training loss 1.716
    Epoch 1, batch 117, training loss 1.718
    Epoch 1, batch 118, training loss 1.720
    Epoch 1, batch 119, training loss 1.723
    Epoch 1, batch 120, training loss 1.730
    Epoch 1, batch 121, training loss 1.737
    Epoch 1, batch 122, training loss 1.737
    Epoch 1, batch 123, training loss 1.732
    Epoch 1, batch 124, training loss 1.731
    Epoch 1, batch 125, training loss 1.724
    Epoch 1, batch 126, training loss 1.716
    Epoch 1, batch 127, training loss 1.717
    Epoch 1, batch 128, training loss 1.717
    Epoch 1, batch 129, training loss 1.712
    Epoch 1, batch 130, training loss 1.711
    Epoch 1, batch 131, training loss 1.722
    Epoch 1, batch 132, training loss 1.731
    Epoch 1, batch 133, training loss 1.729
    Epoch 1, batch 134, training loss 1.721
    Epoch 1, batch 135, training loss 1.724
    Epoch 1, batch 136, training loss 1.722
    Epoch 1, batch 137, training loss 1.721
    Epoch 1, batch 138, training loss 1.727
    Epoch 1, batch 139, training loss 1.722
    Epoch 1, batch 140, training loss 1.725
    Epoch 1, batch 141, training loss 1.723
    Epoch 1, batch 142, training loss 1.725
    Epoch 1, batch 143, training loss 1.722
    Epoch 1, batch 144, training loss 1.721
    Epoch 1, batch 145, training loss 1.721
    Epoch 1, batch 146, training loss 1.720
    Epoch 1, batch 147, training loss 1.725
    Epoch 1, batch 148, training loss 1.731
    Epoch 1, batch 149, training loss 1.730
    Epoch 1, batch 150, training loss 1.728
    Epoch 1, batch 151, training loss 1.727
    Epoch 1, batch 152, training loss 1.734
    Epoch 1, batch 153, training loss 1.732
    Epoch 1, batch 154, training loss 1.729
    Epoch 1, batch 155, training loss 1.731
    Epoch 1, batch 156, training loss 1.732
    Epoch 1, batch 157, training loss 1.728
    Epoch 1, batch 158, training loss 1.725
    Epoch 1, batch 159, training loss 1.729
    Epoch 1, batch 160, training loss 1.726
    Epoch 1, batch 161, training loss 1.725
    Epoch 1, batch 162, training loss 1.733
    Epoch 1, batch 163, training loss 1.728
    Epoch 1, batch 164, training loss 1.732
    Epoch 1, batch 165, training loss 1.733
    Epoch 1, batch 166, training loss 1.731
    Epoch 1, batch 167, training loss 1.726
    Epoch 1, batch 168, training loss 1.729
    Epoch 1, batch 169, training loss 1.725
    Epoch 1, batch 170, training loss 1.724
    Epoch 1, batch 171, training loss 1.721
    Epoch 1, batch 172, training loss 1.718
    Epoch 1, batch 173, training loss 1.717
    Epoch 1, batch 174, training loss 1.724
    Epoch 1, batch 175, training loss 1.730
    Epoch 1, batch 176, training loss 1.727
    Epoch 1, batch 177, training loss 1.723
    Epoch 1, batch 178, training loss 1.727
    Epoch 1, batch 179, training loss 1.726
    Epoch 1, batch 180, training loss 1.730
    Epoch 1, batch 181, training loss 1.729
    Epoch 2, batch 0, training loss 1.958
    Epoch 2, batch 1, training loss 1.891
    Epoch 2, batch 2, training loss 1.864
    Epoch 2, batch 3, training loss 1.713
    Epoch 2, batch 4, training loss 1.557
    Epoch 2, batch 5, training loss 1.469
    Epoch 2, batch 6, training loss 1.409
    Epoch 2, batch 7, training loss 1.470
    Epoch 2, batch 8, training loss 1.593
    Epoch 2, batch 9, training loss 1.688
    Epoch 2, batch 10, training loss 1.608
    Epoch 2, batch 11, training loss 1.595
    Epoch 2, batch 12, training loss 1.624
    Epoch 2, batch 13, training loss 1.656
    Epoch 2, batch 14, training loss 1.699
    Epoch 2, batch 15, training loss 1.708
    Epoch 2, batch 16, training loss 1.754
    Epoch 2, batch 17, training loss 1.745
    Epoch 2, batch 18, training loss 1.728
    Epoch 2, batch 19, training loss 1.733
    Epoch 2, batch 20, training loss 1.701
    Epoch 2, batch 21, training loss 1.684
    Epoch 2, batch 22, training loss 1.705
    Epoch 2, batch 23, training loss 1.713
    Epoch 2, batch 24, training loss 1.686
    Epoch 2, batch 25, training loss 1.680
    Epoch 2, batch 26, training loss 1.691
    Epoch 2, batch 27, training loss 1.681
    Epoch 2, batch 28, training loss 1.701
    Epoch 2, batch 29, training loss 1.702
    Epoch 2, batch 30, training loss 1.706
    Epoch 2, batch 31, training loss 1.736
    Epoch 2, batch 32, training loss 1.722
    Epoch 2, batch 33, training loss 1.699
    Epoch 2, batch 34, training loss 1.697
    Epoch 2, batch 35, training loss 1.671
    Epoch 2, batch 36, training loss 1.691
    Epoch 2, batch 37, training loss 1.676
    Epoch 2, batch 38, training loss 1.661
    Epoch 2, batch 39, training loss 1.673
    Epoch 2, batch 40, training loss 1.695
    Epoch 2, batch 41, training loss 1.693
    Epoch 2, batch 42, training loss 1.709
    Epoch 2, batch 43, training loss 1.705
    Epoch 2, batch 44, training loss 1.713
    Epoch 2, batch 45, training loss 1.708
    Epoch 2, batch 46, training loss 1.709
    Epoch 2, batch 47, training loss 1.697
    Epoch 2, batch 48, training loss 1.707
    Epoch 2, batch 49, training loss 1.695
    Epoch 2, batch 50, training loss 1.721
    Epoch 2, batch 51, training loss 1.721
    Epoch 2, batch 52, training loss 1.726
    Epoch 2, batch 53, training loss 1.712
    Epoch 2, batch 54, training loss 1.702
    Epoch 2, batch 55, training loss 1.696
    Epoch 2, batch 56, training loss 1.687
    Epoch 2, batch 57, training loss 1.691
    Epoch 2, batch 58, training loss 1.679
    Epoch 2, batch 59, training loss 1.667
    Epoch 2, batch 60, training loss 1.666
    Epoch 2, batch 61, training loss 1.677
    Epoch 2, batch 62, training loss 1.659
    Epoch 2, batch 63, training loss 1.660
    Epoch 2, batch 64, training loss 1.652
    Epoch 2, batch 65, training loss 1.657
    Epoch 2, batch 66, training loss 1.658
    Epoch 2, batch 67, training loss 1.668
    Epoch 2, batch 68, training loss 1.656
    Epoch 2, batch 69, training loss 1.650
    Epoch 2, batch 70, training loss 1.666
    Epoch 2, batch 71, training loss 1.664
    Epoch 2, batch 72, training loss 1.651
    Epoch 2, batch 73, training loss 1.665
    Epoch 2, batch 74, training loss 1.660
    Epoch 2, batch 75, training loss 1.661
    Epoch 2, batch 76, training loss 1.654
    Epoch 2, batch 77, training loss 1.671
    Epoch 2, batch 78, training loss 1.668
    Epoch 2, batch 79, training loss 1.664
    Epoch 2, batch 80, training loss 1.666
    Epoch 2, batch 81, training loss 1.659
    Epoch 2, batch 82, training loss 1.656
    Epoch 2, batch 83, training loss 1.651
    Epoch 2, batch 84, training loss 1.658
    Epoch 2, batch 85, training loss 1.657
    Epoch 2, batch 86, training loss 1.656
    Epoch 2, batch 87, training loss 1.656
    Epoch 2, batch 88, training loss 1.650
    Epoch 2, batch 89, training loss 1.648
    Epoch 2, batch 90, training loss 1.639
    Epoch 2, batch 91, training loss 1.650
    Epoch 2, batch 92, training loss 1.655
    Epoch 2, batch 93, training loss 1.651
    Epoch 2, batch 94, training loss 1.658
    Epoch 2, batch 95, training loss 1.661
    Epoch 2, batch 96, training loss 1.674
    Epoch 2, batch 97, training loss 1.673
    Epoch 2, batch 98, training loss 1.675
    Epoch 2, batch 99, training loss 1.670
    Epoch 2, batch 100, training loss 1.668
    Epoch 2, batch 101, training loss 1.661
    Epoch 2, batch 102, training loss 1.667
    Epoch 2, batch 103, training loss 1.667
    Epoch 2, batch 104, training loss 1.658
    Epoch 2, batch 105, training loss 1.659
    Epoch 2, batch 106, training loss 1.657
    Epoch 2, batch 107, training loss 1.657
    Epoch 2, batch 108, training loss 1.649
    Epoch 2, batch 109, training loss 1.644
    Epoch 2, batch 110, training loss 1.641
    Epoch 2, batch 111, training loss 1.640
    Epoch 2, batch 112, training loss 1.638
    Epoch 2, batch 113, training loss 1.646
    Epoch 2, batch 114, training loss 1.645
    Epoch 2, batch 115, training loss 1.645
    Epoch 2, batch 116, training loss 1.641
    Epoch 2, batch 117, training loss 1.643
    Epoch 2, batch 118, training loss 1.648
    Epoch 2, batch 119, training loss 1.650
    Epoch 2, batch 120, training loss 1.646
    Epoch 2, batch 121, training loss 1.649
    Epoch 2, batch 122, training loss 1.649
    Epoch 2, batch 123, training loss 1.650
    Epoch 2, batch 124, training loss 1.650
    Epoch 2, batch 125, training loss 1.643
    Epoch 2, batch 126, training loss 1.650
    Epoch 2, batch 127, training loss 1.648
    Epoch 2, batch 128, training loss 1.642
    Epoch 2, batch 129, training loss 1.642
    Epoch 2, batch 130, training loss 1.643
    Epoch 2, batch 131, training loss 1.636
    Epoch 2, batch 132, training loss 1.635
    Epoch 2, batch 133, training loss 1.645
    Epoch 2, batch 134, training loss 1.642
    Epoch 2, batch 135, training loss 1.647
    Epoch 2, batch 136, training loss 1.646
    Epoch 2, batch 137, training loss 1.655
    Epoch 2, batch 138, training loss 1.652
    Epoch 2, batch 139, training loss 1.647
    Epoch 2, batch 140, training loss 1.645
    Epoch 2, batch 141, training loss 1.642
    Epoch 2, batch 142, training loss 1.637
    Epoch 2, batch 143, training loss 1.635
    Epoch 2, batch 144, training loss 1.635
    Epoch 2, batch 145, training loss 1.635
    Epoch 2, batch 146, training loss 1.633
    Epoch 2, batch 147, training loss 1.628
    Epoch 2, batch 148, training loss 1.625
    Epoch 2, batch 149, training loss 1.622
    Epoch 2, batch 150, training loss 1.624
    Epoch 2, batch 151, training loss 1.622
    Epoch 2, batch 152, training loss 1.621
    Epoch 2, batch 153, training loss 1.618
    Epoch 2, batch 154, training loss 1.618
    Epoch 2, batch 155, training loss 1.621
    Epoch 2, batch 156, training loss 1.619
    Epoch 2, batch 157, training loss 1.618
    Epoch 2, batch 158, training loss 1.615
    Epoch 2, batch 159, training loss 1.615
    Epoch 2, batch 160, training loss 1.618
    Epoch 2, batch 161, training loss 1.618
    Epoch 2, batch 162, training loss 1.614
    Epoch 2, batch 163, training loss 1.616
    Epoch 2, batch 164, training loss 1.614
    Epoch 2, batch 165, training loss 1.609
    Epoch 2, batch 166, training loss 1.607
    Epoch 2, batch 167, training loss 1.605
    Epoch 2, batch 168, training loss 1.599
    Epoch 2, batch 169, training loss 1.598
    Epoch 2, batch 170, training loss 1.599
    Epoch 2, batch 171, training loss 1.606
    Epoch 2, batch 172, training loss 1.606
    Epoch 2, batch 173, training loss 1.608
    Epoch 2, batch 174, training loss 1.610
    Epoch 2, batch 175, training loss 1.611
    Epoch 2, batch 176, training loss 1.609
    Epoch 2, batch 177, training loss 1.610
    Epoch 2, batch 178, training loss 1.608
    Epoch 2, batch 179, training loss 1.605
    Epoch 2, batch 180, training loss 1.603
    Epoch 2, batch 181, training loss 1.606
    Epoch 3, batch 0, training loss 0.897
    Epoch 3, batch 1, training loss 1.449
    Epoch 3, batch 2, training loss 1.686
    Epoch 3, batch 3, training loss 1.557
    Epoch 3, batch 4, training loss 1.550
    Epoch 3, batch 5, training loss 1.588
    Epoch 3, batch 6, training loss 1.544
    Epoch 3, batch 7, training loss 1.576
    Epoch 3, batch 8, training loss 1.650
    Epoch 3, batch 9, training loss 1.650
    Epoch 3, batch 10, training loss 1.739
    Epoch 3, batch 11, training loss 1.694
    Epoch 3, batch 12, training loss 1.793
    Epoch 3, batch 13, training loss 1.806
    Epoch 3, batch 14, training loss 1.750
    Epoch 3, batch 15, training loss 1.800
    Epoch 3, batch 16, training loss 1.766
    Epoch 3, batch 17, training loss 1.713
    Epoch 3, batch 18, training loss 1.668
    Epoch 3, batch 19, training loss 1.675
    Epoch 3, batch 20, training loss 1.670
    Epoch 3, batch 21, training loss 1.658
    Epoch 3, batch 22, training loss 1.629
    Epoch 3, batch 23, training loss 1.627
    Epoch 3, batch 24, training loss 1.598
    Epoch 3, batch 25, training loss 1.578
    Epoch 3, batch 26, training loss 1.619
    Epoch 3, batch 27, training loss 1.654
    Epoch 3, batch 28, training loss 1.628
    Epoch 3, batch 29, training loss 1.679
    Epoch 3, batch 30, training loss 1.663
    Epoch 3, batch 31, training loss 1.711
    Epoch 3, batch 32, training loss 1.742
    Epoch 3, batch 33, training loss 1.736
    Epoch 3, batch 34, training loss 1.725
    Epoch 3, batch 35, training loss 1.704
    Epoch 3, batch 36, training loss 1.689
    Epoch 3, batch 37, training loss 1.674
    Epoch 3, batch 38, training loss 1.663
    Epoch 3, batch 39, training loss 1.647
    Epoch 3, batch 40, training loss 1.661
    Epoch 3, batch 41, training loss 1.665
    Epoch 3, batch 42, training loss 1.660
    Epoch 3, batch 43, training loss 1.648
    Epoch 3, batch 44, training loss 1.643
    Epoch 3, batch 45, training loss 1.627
    Epoch 3, batch 46, training loss 1.635
    Epoch 3, batch 47, training loss 1.619
    Epoch 3, batch 48, training loss 1.623
    Epoch 3, batch 49, training loss 1.624
    Epoch 3, batch 50, training loss 1.612
    Epoch 3, batch 51, training loss 1.627
    Epoch 3, batch 52, training loss 1.637
    Epoch 3, batch 53, training loss 1.635
    Epoch 3, batch 54, training loss 1.630
    Epoch 3, batch 55, training loss 1.628
    Epoch 3, batch 56, training loss 1.633
    Epoch 3, batch 57, training loss 1.641
    Epoch 3, batch 58, training loss 1.629
    Epoch 3, batch 59, training loss 1.635
    Epoch 3, batch 60, training loss 1.627
    Epoch 3, batch 61, training loss 1.634
    Epoch 3, batch 62, training loss 1.624
    Epoch 3, batch 63, training loss 1.624
    Epoch 3, batch 64, training loss 1.634
    Epoch 3, batch 65, training loss 1.624
    Epoch 3, batch 66, training loss 1.612
    Epoch 3, batch 67, training loss 1.626
    Epoch 3, batch 68, training loss 1.635
    Epoch 3, batch 69, training loss 1.620
    Epoch 3, batch 70, training loss 1.619
    Epoch 3, batch 71, training loss 1.622
    Epoch 3, batch 72, training loss 1.617
    Epoch 3, batch 73, training loss 1.616
    Epoch 3, batch 74, training loss 1.620
    Epoch 3, batch 75, training loss 1.619
    Epoch 3, batch 76, training loss 1.627
    Epoch 3, batch 77, training loss 1.619
    Epoch 3, batch 78, training loss 1.610
    Epoch 3, batch 79, training loss 1.602
    Epoch 3, batch 80, training loss 1.599
    Epoch 3, batch 81, training loss 1.601
    Epoch 3, batch 82, training loss 1.608
    Epoch 3, batch 83, training loss 1.611
    Epoch 3, batch 84, training loss 1.617
    Epoch 3, batch 85, training loss 1.612
    Epoch 3, batch 86, training loss 1.615
    Epoch 3, batch 87, training loss 1.609
    Epoch 3, batch 88, training loss 1.609
    Epoch 3, batch 89, training loss 1.609
    Epoch 3, batch 90, training loss 1.620
    Epoch 3, batch 91, training loss 1.626
    Epoch 3, batch 92, training loss 1.628
    Epoch 3, batch 93, training loss 1.625
    Epoch 3, batch 94, training loss 1.621
    Epoch 3, batch 95, training loss 1.625
    Epoch 3, batch 96, training loss 1.626
    Epoch 3, batch 97, training loss 1.628
    Epoch 3, batch 98, training loss 1.641
    Epoch 3, batch 99, training loss 1.641
    Epoch 3, batch 100, training loss 1.635
    Epoch 3, batch 101, training loss 1.629
    Epoch 3, batch 102, training loss 1.622
    Epoch 3, batch 103, training loss 1.617
    Epoch 3, batch 104, training loss 1.621
    Epoch 3, batch 105, training loss 1.625
    Epoch 3, batch 106, training loss 1.629
    Epoch 3, batch 107, training loss 1.623
    Epoch 3, batch 108, training loss 1.616
    Epoch 3, batch 109, training loss 1.620
    Epoch 3, batch 110, training loss 1.618
    Epoch 3, batch 111, training loss 1.619
    Epoch 3, batch 112, training loss 1.611
    Epoch 3, batch 113, training loss 1.606
    Epoch 3, batch 114, training loss 1.604
    Epoch 3, batch 115, training loss 1.600
    Epoch 3, batch 116, training loss 1.602
    Epoch 3, batch 117, training loss 1.604
    Epoch 3, batch 118, training loss 1.603
    Epoch 3, batch 119, training loss 1.597
    Epoch 3, batch 120, training loss 1.594
    Epoch 3, batch 121, training loss 1.598
    Epoch 3, batch 122, training loss 1.606
    Epoch 3, batch 123, training loss 1.614
    Epoch 3, batch 124, training loss 1.611
    Epoch 3, batch 125, training loss 1.605
    Epoch 3, batch 126, training loss 1.601
    Epoch 3, batch 127, training loss 1.595
    Epoch 3, batch 128, training loss 1.594
    Epoch 3, batch 129, training loss 1.602
    Epoch 3, batch 130, training loss 1.604
    Epoch 3, batch 131, training loss 1.602
    Epoch 3, batch 132, training loss 1.600
    Epoch 3, batch 133, training loss 1.605
    Epoch 3, batch 134, training loss 1.599
    Epoch 3, batch 135, training loss 1.594
    Epoch 3, batch 136, training loss 1.591
    Epoch 3, batch 137, training loss 1.594
    Epoch 3, batch 138, training loss 1.592
    Epoch 3, batch 139, training loss 1.592
    Epoch 3, batch 140, training loss 1.590
    Epoch 3, batch 141, training loss 1.585
    Epoch 3, batch 142, training loss 1.585
    Epoch 3, batch 143, training loss 1.594
    Epoch 3, batch 144, training loss 1.591
    Epoch 3, batch 145, training loss 1.584
    Epoch 3, batch 146, training loss 1.587
    Epoch 3, batch 147, training loss 1.588
    Epoch 3, batch 148, training loss 1.590
    Epoch 3, batch 149, training loss 1.587
    Epoch 3, batch 150, training loss 1.581
    Epoch 3, batch 151, training loss 1.580
    Epoch 3, batch 152, training loss 1.583
    Epoch 3, batch 153, training loss 1.585
    Epoch 3, batch 154, training loss 1.590
    Epoch 3, batch 155, training loss 1.589
    Epoch 3, batch 156, training loss 1.593
    Epoch 3, batch 157, training loss 1.600
    Epoch 3, batch 158, training loss 1.599
    Epoch 3, batch 159, training loss 1.600
    Epoch 3, batch 160, training loss 1.602
    Epoch 3, batch 161, training loss 1.601
    Epoch 3, batch 162, training loss 1.604
    Epoch 3, batch 163, training loss 1.603
    Epoch 3, batch 164, training loss 1.599
    Epoch 3, batch 165, training loss 1.595
    Epoch 3, batch 166, training loss 1.593
    Epoch 3, batch 167, training loss 1.594
    Epoch 3, batch 168, training loss 1.594
    Epoch 3, batch 169, training loss 1.598
    Epoch 3, batch 170, training loss 1.597
    Epoch 3, batch 171, training loss 1.597
    Epoch 3, batch 172, training loss 1.601
    Epoch 3, batch 173, training loss 1.597
    Epoch 3, batch 174, training loss 1.591
    Epoch 3, batch 175, training loss 1.599
    Epoch 3, batch 176, training loss 1.599
    Epoch 3, batch 177, training loss 1.601
    Epoch 3, batch 178, training loss 1.600
    Epoch 3, batch 179, training loss 1.599
    Epoch 3, batch 180, training loss 1.600
    Epoch 3, batch 181, training loss 1.601
    Epoch 4, batch 0, training loss 1.138
    Epoch 4, batch 1, training loss 1.059
    Epoch 4, batch 2, training loss 1.126
    Epoch 4, batch 3, training loss 1.143
    Epoch 4, batch 4, training loss 1.205
    Epoch 4, batch 5, training loss 1.179
    Epoch 4, batch 6, training loss 1.164
    Epoch 4, batch 7, training loss 1.179
    Epoch 4, batch 8, training loss 1.226
    Epoch 4, batch 9, training loss 1.430
    Epoch 4, batch 10, training loss 1.457
    Epoch 4, batch 11, training loss 1.528
    Epoch 4, batch 12, training loss 1.487
    Epoch 4, batch 13, training loss 1.537
    Epoch 4, batch 14, training loss 1.513
    Epoch 4, batch 15, training loss 1.483
    Epoch 4, batch 16, training loss 1.463
    Epoch 4, batch 17, training loss 1.431
    Epoch 4, batch 18, training loss 1.421
    Epoch 4, batch 19, training loss 1.395
    Epoch 4, batch 20, training loss 1.407
    Epoch 4, batch 21, training loss 1.384
    Epoch 4, batch 22, training loss 1.385
    Epoch 4, batch 23, training loss 1.455
    Epoch 4, batch 24, training loss 1.445
    Epoch 4, batch 25, training loss 1.447
    Epoch 4, batch 26, training loss 1.432
    Epoch 4, batch 27, training loss 1.423
    Epoch 4, batch 28, training loss 1.427
    Epoch 4, batch 29, training loss 1.443
    Epoch 4, batch 30, training loss 1.454
    Epoch 4, batch 31, training loss 1.433
    Epoch 4, batch 32, training loss 1.438
    Epoch 4, batch 33, training loss 1.466
    Epoch 4, batch 34, training loss 1.472
    Epoch 4, batch 35, training loss 1.497
    Epoch 4, batch 36, training loss 1.487
    Epoch 4, batch 37, training loss 1.507
    Epoch 4, batch 38, training loss 1.501
    Epoch 4, batch 39, training loss 1.488
    Epoch 4, batch 40, training loss 1.498
    Epoch 4, batch 41, training loss 1.483
    Epoch 4, batch 42, training loss 1.486
    Epoch 4, batch 43, training loss 1.502
    Epoch 4, batch 44, training loss 1.495
    Epoch 4, batch 45, training loss 1.500
    Epoch 4, batch 46, training loss 1.489
    Epoch 4, batch 47, training loss 1.477
    Epoch 4, batch 48, training loss 1.469
    Epoch 4, batch 49, training loss 1.482
    Epoch 4, batch 50, training loss 1.478
    Epoch 4, batch 51, training loss 1.471
    Epoch 4, batch 52, training loss 1.481
    Epoch 4, batch 53, training loss 1.487
    Epoch 4, batch 54, training loss 1.474
    Epoch 4, batch 55, training loss 1.475
    Epoch 4, batch 56, training loss 1.462
    Epoch 4, batch 57, training loss 1.464
    Epoch 4, batch 58, training loss 1.462
    Epoch 4, batch 59, training loss 1.470
    Epoch 4, batch 60, training loss 1.480
    Epoch 4, batch 61, training loss 1.471
    Epoch 4, batch 62, training loss 1.471
    Epoch 4, batch 63, training loss 1.462
    Epoch 4, batch 64, training loss 1.484
    Epoch 4, batch 65, training loss 1.474
    Epoch 4, batch 66, training loss 1.469
    Epoch 4, batch 67, training loss 1.478
    Epoch 4, batch 68, training loss 1.476
    Epoch 4, batch 69, training loss 1.465
    Epoch 4, batch 70, training loss 1.462
    Epoch 4, batch 71, training loss 1.453
    Epoch 4, batch 72, training loss 1.465
    Epoch 4, batch 73, training loss 1.465
    Epoch 4, batch 74, training loss 1.464
    Epoch 4, batch 75, training loss 1.460
    Epoch 4, batch 76, training loss 1.457
    Epoch 4, batch 77, training loss 1.455
    Epoch 4, batch 78, training loss 1.446
    Epoch 4, batch 79, training loss 1.446
    Epoch 4, batch 80, training loss 1.441
    Epoch 4, batch 81, training loss 1.464
    Epoch 4, batch 82, training loss 1.456
    Epoch 4, batch 83, training loss 1.460
    Epoch 4, batch 84, training loss 1.452
    Epoch 4, batch 85, training loss 1.448
    Epoch 4, batch 86, training loss 1.460
    Epoch 4, batch 87, training loss 1.479
    Epoch 4, batch 88, training loss 1.475
    Epoch 4, batch 89, training loss 1.475
    Epoch 4, batch 90, training loss 1.471
    Epoch 4, batch 91, training loss 1.472
    Epoch 4, batch 92, training loss 1.477
    Epoch 4, batch 93, training loss 1.475
    Epoch 4, batch 94, training loss 1.472
    Epoch 4, batch 95, training loss 1.484
    Epoch 4, batch 96, training loss 1.482
    Epoch 4, batch 97, training loss 1.474
    Epoch 4, batch 98, training loss 1.483
    Epoch 4, batch 99, training loss 1.495
    Epoch 4, batch 100, training loss 1.495
    Epoch 4, batch 101, training loss 1.507
    Epoch 4, batch 102, training loss 1.501
    Epoch 4, batch 103, training loss 1.498
    Epoch 4, batch 104, training loss 1.496
    Epoch 4, batch 105, training loss 1.491
    Epoch 4, batch 106, training loss 1.486
    Epoch 4, batch 107, training loss 1.486
    Epoch 4, batch 108, training loss 1.491
    Epoch 4, batch 109, training loss 1.493
    Epoch 4, batch 110, training loss 1.490
    Epoch 4, batch 111, training loss 1.483
    Epoch 4, batch 112, training loss 1.488
    Epoch 4, batch 113, training loss 1.492
    Epoch 4, batch 114, training loss 1.487
    Epoch 4, batch 115, training loss 1.495
    Epoch 4, batch 116, training loss 1.507
    Epoch 4, batch 117, training loss 1.501
    Epoch 4, batch 118, training loss 1.507
    Epoch 4, batch 119, training loss 1.508
    Epoch 4, batch 120, training loss 1.510
    Epoch 4, batch 121, training loss 1.521
    Epoch 4, batch 122, training loss 1.518
    Epoch 4, batch 123, training loss 1.520
    Epoch 4, batch 124, training loss 1.525
    Epoch 4, batch 125, training loss 1.523
    Epoch 4, batch 126, training loss 1.524
    Epoch 4, batch 127, training loss 1.526
    Epoch 4, batch 128, training loss 1.523
    Epoch 4, batch 129, training loss 1.524
    Epoch 4, batch 130, training loss 1.528
    Epoch 4, batch 131, training loss 1.526
    Epoch 4, batch 132, training loss 1.527
    Epoch 4, batch 133, training loss 1.525
    Epoch 4, batch 134, training loss 1.522
    Epoch 4, batch 135, training loss 1.524
    Epoch 4, batch 136, training loss 1.522
    Epoch 4, batch 137, training loss 1.519
    Epoch 4, batch 138, training loss 1.521
    Epoch 4, batch 139, training loss 1.523
    Epoch 4, batch 140, training loss 1.525
    Epoch 4, batch 141, training loss 1.520
    Epoch 4, batch 142, training loss 1.521
    Epoch 4, batch 143, training loss 1.517
    Epoch 4, batch 144, training loss 1.521
    Epoch 4, batch 145, training loss 1.523
    Epoch 4, batch 146, training loss 1.521
    Epoch 4, batch 147, training loss 1.523
    Epoch 4, batch 148, training loss 1.521
    Epoch 4, batch 149, training loss 1.524
    Epoch 4, batch 150, training loss 1.526
    Epoch 4, batch 151, training loss 1.523
    Epoch 4, batch 152, training loss 1.519
    Epoch 4, batch 153, training loss 1.517
    Epoch 4, batch 154, training loss 1.518
    Epoch 4, batch 155, training loss 1.522
    Epoch 4, batch 156, training loss 1.521
    Epoch 4, batch 157, training loss 1.519
    Epoch 4, batch 158, training loss 1.521
    Epoch 4, batch 159, training loss 1.519
    Epoch 4, batch 160, training loss 1.513
    Epoch 4, batch 161, training loss 1.514
    Epoch 4, batch 162, training loss 1.515
    Epoch 4, batch 163, training loss 1.508
    Epoch 4, batch 164, training loss 1.510
    Epoch 4, batch 165, training loss 1.511
    Epoch 4, batch 166, training loss 1.507
    Epoch 4, batch 167, training loss 1.505
    Epoch 4, batch 168, training loss 1.501
    Epoch 4, batch 169, training loss 1.497
    Epoch 4, batch 170, training loss 1.497
    Epoch 4, batch 171, training loss 1.494
    Epoch 4, batch 172, training loss 1.493
    Epoch 4, batch 173, training loss 1.492
    Epoch 4, batch 174, training loss 1.489
    Epoch 4, batch 175, training loss 1.490
    Epoch 4, batch 176, training loss 1.490
    Epoch 4, batch 177, training loss 1.488
    Epoch 4, batch 178, training loss 1.487
    Epoch 4, batch 179, training loss 1.488
    Epoch 4, batch 180, training loss 1.485
    Epoch 4, batch 181, training loss 1.482


Let's re-predict on the image we used.


```python
input_img = image.imread('example.jpg')
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
img = transform_fn(input_img)
img = img.expand_dims(0).as_in_context(mx.gpu(0))

output = net.module.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
mask = viz.get_color_pallete(predict, 'pascal_voc')
mask = np.array(mask.convert('RGB'), dtype=np.int)

combined = (mask+input_img.asnumpy())/2
plt.imshow(combined.astype(np.uint8))
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/SemanticSegmentation_files/SemanticSegmentation_46_0.png)


Although this is a small example, it is basically how we train a model on much larger dataset.

## Further resources

On the [GluonCV Segmentation Model Zoo](https://gluon-cv.mxnet.io/model_zoo/segmentation.html) page, we provide:

- Training scripts for FCN/PSPnet/DeepLab/Mask R-CNN on MS COCO/Pascal VOC/Cityscapes.
- Training hyperparameters to reproduce.
- Training Logs to compare speed and accuracy.
