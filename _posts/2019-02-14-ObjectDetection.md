---
layout: post
title: Object Detection
---

## Configuration

```python
# !pip install mxnet-cu92 --pre --upgrade
# !pip install gluoncv --pre
```


```python
import time
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon, nd, image
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.utils import download, viz
from gluoncv.model_zoo import get_model
from gluoncv.utils import viz, download
```

## Classification with multiple objects

Classification does an excellent job for images with a main object. However, what if the input photo contains multiple objects?


```python
plt.rcParams['figure.figsize'] = (15, 9)

img = image.imread('cr7.jpg')
viz.plot_image(img)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_3_0.png)


We predict with image classification algorithm.


```python
model_name = 'ResNet50_v1'
net = gluoncv.model_zoo.get_model(model_name, pretrained=True)

transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)

pred = net(transformed_img)
prob = mx.nd.softmax(pred)[0].asnumpy()
ind = mx.nd.topk(pred, k=5)[0].astype('int').asnumpy().tolist()
print('The input picture is classified to be')
for i in range(5):
    print('- [%s], with probability %.3f.'%(net.classes[ind[i]], prob[ind[i]]))
```

    The input picture is classified to be
    - [soccer ball], with probability 0.434.
    - [ballplayer], with probability 0.203.
    - [basketball], with probability 0.027.
    - [baseball], with probability 0.027.
    - [rugby ball], with probability 0.022.


The model's best guess is just "soccer ball". This is not how we describe the image.

## Try Object Detection

Next, let's try to use object detection to detect the multiple objects.

Firstly we again load the image in and preprocess it at the same time. Here the `short=512` means we resize the short edge to 512 pixels while keeping the aspect ratio.


```python
x, img = gluoncv.data.transforms.presets.ssd.load_test('cr7.jpg', short=512)
viz.plot_image(img)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_7_0.png)


Next, we download a pretraiend SSD. Here `ssd_512_resnet50_v1_coco` is a combination of:

1. `ssd`: the name of the model
2. `512`: the length of the short edge
3. `resnet50_v1`: the base model
4. `coco`: the dataset on which the model is trained


```python
net = get_model('ssd_512_resnet50_v1_coco', pretrained=True)
```

The output contains the predicted classes, the confident scores and the location of bounding boxes.

With `gluoncv.utils.viz` we can easily visualize the result.

One interesting finding is that the algorithm can even detect the blurred out audiences in the background. Although their scores are lower.


```python
class_IDs, scores, bounding_boxs = net(x)
viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_11_0.png)


Another example: this is a photo taken in our Palo Alto office during world cup 2018.

We can detect and count how many are watching the game instead of going back to work.


```python
x, img = gluoncv.data.transforms.presets.ssd.load_test('crowd.png', short=512)
viz.plot_image(img)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_13_0.png)



```python
class_IDs, scores, bounding_boxs = net(x)
viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_14_0.png)


We can select all boxes containing a `person` count.


```python
class_IDs, scores, bounding_boxs = net(x)
person_ind = [i for i, cls in enumerate(net.classes) if cls == 'person']
ind = np.nonzero(class_IDs[0].asnumpy() == person_ind)[0]

new_class_IDs = class_IDs[0][ind]
new_scores = scores[0][ind]
new_bounding_boxs = bounding_boxs[0][ind]

viz.plot_bbox(img, new_bounding_boxs, new_scores, new_class_IDs, class_names=net.classes)
plt.show()

print('There are %d people in this photo.'%(len(ind)))
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_16_0.png)


    There are 33 people in this photo.


Detection can also be used in outdoor scenario. For example this is a photo of our Palo Alto office.


```python
x, img = gluoncv.data.transforms.presets.ssd.load_test('streetview_amazon.png', short=512)
viz.plot_image(img)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_18_0.png)


We can detect the cars, trucks and traffic lights.

Guess where can we apply it to? Automated Driving!


```python
class_IDs, scores, bounding_boxs = net(x)
viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_20_0.png)


## Train your own model to detect Pikachu

Pre-trained models only recognize the type of objects in the dataset. What if you have your own object of interests and want to train a model for that?

No problem! We'll show an example of teaching a pre-trained model to recognize Pikachu.

First we download the small dataset.


```python
url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.rec'
idx_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.idx'
download(url, path='pikachu_train.rec', overwrite=False)
download(idx_url, path='pikachu_train.idx', overwrite=False)
```




    'pikachu_train.idx'



Next we load in the dataset and show how it looks like.


```python
dataset = gluoncv.data.RecordFileDetection('pikachu_train.rec')
classes = ['pikachu']  # only one foreground class here
image, label = dataset[1]
print('label:', label)
# display image and label
ax = viz.plot_bbox(image, bboxes=label[:, :4], labels=label[:, 4:5], class_names=classes)
plt.show()
```

    label: [[214.70685 132.91504 271.07706 215.32448   0.     ]]



![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_24_1.png)


Basically we chose some landscape photos and copy an open-sourced Pikachu figure onto the image.

We load in a pretrained `ssd_512_mobilenet1.0_voc` model and reset its output layer.


```python
net = gluoncv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
net.reset_class(classes)
net = gluoncv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes,
    pretrained_base=False, transfer='voc')
```

Next we define the dataloader.


```python
def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader

train_data = get_dataloader(net, dataset, 512, 16, 0)
```

We encourage to use GPU for the training.


```python
ctx = [mx.gpu(0)]
net.collect_params().reset_ctx(ctx)
```

Next we define the trainer, loss and metric.


```python
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})
mbox_loss = gluoncv.loss.SSDMultiBoxLoss()
ce_metric = mx.metric.Loss('CrossEntropy')
smoothl1_metric = mx.metric.Loss('SmoothL1')
```

Everything's prepared. We can start training now. We just need 2 epochs for this dataset.


```python
for epoch in range(0, 2):
    ce_metric.reset()
    smoothl1_metric.reset()
    tic = time.time()
    btic = time.time()
    net.hybridize(static_alloc=True, static_shape=True)
    for i, batch in enumerate(train_data):
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
            autograd.backward(sum_loss)
        # since we have already normalized the loss, we don't want to normalize
        # by batch-size anymore
        trainer.step(1)
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])
        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        if i % 1 == 0:
            print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
        btic = time.time()

net.save_parameters('ssd_512_mobilenet1.0_pikachu.params')
```

    [Epoch 0][Batch 0], Speed: 3.187 samples/sec, CrossEntropy=7.848, SmoothL1=1.758
    [Epoch 0][Batch 1], Speed: 23.103 samples/sec, CrossEntropy=7.009, SmoothL1=2.131
    [Epoch 0][Batch 2], Speed: 26.272 samples/sec, CrossEntropy=6.494, SmoothL1=1.996
    [Epoch 0][Batch 3], Speed: 23.810 samples/sec, CrossEntropy=6.144, SmoothL1=1.942
    [Epoch 0][Batch 4], Speed: 21.310 samples/sec, CrossEntropy=5.837, SmoothL1=1.891
    [Epoch 0][Batch 5], Speed: 24.496 samples/sec, CrossEntropy=5.606, SmoothL1=1.877
    [Epoch 0][Batch 6], Speed: 21.242 samples/sec, CrossEntropy=5.402, SmoothL1=1.846
    [Epoch 0][Batch 7], Speed: 24.172 samples/sec, CrossEntropy=5.205, SmoothL1=1.744
    [Epoch 0][Batch 8], Speed: 23.632 samples/sec, CrossEntropy=5.041, SmoothL1=1.663
    [Epoch 0][Batch 9], Speed: 25.073 samples/sec, CrossEntropy=4.880, SmoothL1=1.605
    [Epoch 0][Batch 10], Speed: 20.212 samples/sec, CrossEntropy=4.725, SmoothL1=1.552
    [Epoch 0][Batch 11], Speed: 21.496 samples/sec, CrossEntropy=4.599, SmoothL1=1.537
    [Epoch 0][Batch 12], Speed: 24.680 samples/sec, CrossEntropy=4.485, SmoothL1=1.513
    [Epoch 0][Batch 13], Speed: 24.742 samples/sec, CrossEntropy=4.396, SmoothL1=1.468
    [Epoch 0][Batch 14], Speed: 24.007 samples/sec, CrossEntropy=4.288, SmoothL1=1.423
    [Epoch 0][Batch 15], Speed: 23.327 samples/sec, CrossEntropy=4.198, SmoothL1=1.410
    [Epoch 0][Batch 16], Speed: 22.524 samples/sec, CrossEntropy=4.109, SmoothL1=1.410
    [Epoch 0][Batch 17], Speed: 24.328 samples/sec, CrossEntropy=4.016, SmoothL1=1.375
    [Epoch 0][Batch 18], Speed: 24.391 samples/sec, CrossEntropy=3.943, SmoothL1=1.341
    [Epoch 0][Batch 19], Speed: 25.809 samples/sec, CrossEntropy=3.870, SmoothL1=1.303
    [Epoch 0][Batch 20], Speed: 24.707 samples/sec, CrossEntropy=3.794, SmoothL1=1.267
    [Epoch 0][Batch 21], Speed: 22.349 samples/sec, CrossEntropy=3.730, SmoothL1=1.235
    [Epoch 0][Batch 22], Speed: 24.374 samples/sec, CrossEntropy=3.666, SmoothL1=1.209
    [Epoch 0][Batch 23], Speed: 24.637 samples/sec, CrossEntropy=3.614, SmoothL1=1.184
    [Epoch 0][Batch 24], Speed: 23.471 samples/sec, CrossEntropy=3.563, SmoothL1=1.169
    [Epoch 0][Batch 25], Speed: 24.125 samples/sec, CrossEntropy=3.506, SmoothL1=1.147
    [Epoch 0][Batch 26], Speed: 21.113 samples/sec, CrossEntropy=3.451, SmoothL1=1.130
    [Epoch 0][Batch 27], Speed: 23.711 samples/sec, CrossEntropy=3.404, SmoothL1=1.116
    [Epoch 0][Batch 28], Speed: 22.144 samples/sec, CrossEntropy=3.357, SmoothL1=1.104
    [Epoch 0][Batch 29], Speed: 20.890 samples/sec, CrossEntropy=3.314, SmoothL1=1.089
    [Epoch 0][Batch 30], Speed: 22.752 samples/sec, CrossEntropy=3.267, SmoothL1=1.074
    [Epoch 0][Batch 31], Speed: 27.698 samples/sec, CrossEntropy=3.232, SmoothL1=1.058
    [Epoch 0][Batch 32], Speed: 23.063 samples/sec, CrossEntropy=3.193, SmoothL1=1.042
    [Epoch 0][Batch 33], Speed: 24.466 samples/sec, CrossEntropy=3.155, SmoothL1=1.029
    [Epoch 0][Batch 34], Speed: 21.853 samples/sec, CrossEntropy=3.121, SmoothL1=1.019
    [Epoch 0][Batch 35], Speed: 22.306 samples/sec, CrossEntropy=3.090, SmoothL1=1.008
    [Epoch 0][Batch 36], Speed: 25.920 samples/sec, CrossEntropy=3.062, SmoothL1=0.999
    [Epoch 0][Batch 37], Speed: 24.774 samples/sec, CrossEntropy=3.030, SmoothL1=0.989
    [Epoch 0][Batch 38], Speed: 22.587 samples/sec, CrossEntropy=3.000, SmoothL1=0.982
    [Epoch 0][Batch 39], Speed: 21.712 samples/sec, CrossEntropy=2.971, SmoothL1=0.967
    [Epoch 0][Batch 40], Speed: 23.676 samples/sec, CrossEntropy=2.945, SmoothL1=0.958
    [Epoch 0][Batch 41], Speed: 24.470 samples/sec, CrossEntropy=2.919, SmoothL1=0.949
    [Epoch 0][Batch 42], Speed: 22.321 samples/sec, CrossEntropy=2.894, SmoothL1=0.946
    [Epoch 0][Batch 43], Speed: 21.651 samples/sec, CrossEntropy=2.870, SmoothL1=0.940
    [Epoch 0][Batch 44], Speed: 24.359 samples/sec, CrossEntropy=2.842, SmoothL1=0.930
    [Epoch 0][Batch 45], Speed: 24.733 samples/sec, CrossEntropy=2.818, SmoothL1=0.921
    [Epoch 0][Batch 46], Speed: 23.699 samples/sec, CrossEntropy=2.795, SmoothL1=0.915
    [Epoch 0][Batch 47], Speed: 21.677 samples/sec, CrossEntropy=2.774, SmoothL1=0.906
    [Epoch 0][Batch 48], Speed: 27.813 samples/sec, CrossEntropy=2.753, SmoothL1=0.898
    [Epoch 0][Batch 49], Speed: 23.635 samples/sec, CrossEntropy=2.728, SmoothL1=0.889
    [Epoch 0][Batch 50], Speed: 24.476 samples/sec, CrossEntropy=2.710, SmoothL1=0.880
    [Epoch 0][Batch 51], Speed: 22.757 samples/sec, CrossEntropy=2.689, SmoothL1=0.871
    [Epoch 0][Batch 52], Speed: 22.535 samples/sec, CrossEntropy=2.669, SmoothL1=0.868
    [Epoch 0][Batch 53], Speed: 24.478 samples/sec, CrossEntropy=2.651, SmoothL1=0.858
    [Epoch 0][Batch 54], Speed: 25.408 samples/sec, CrossEntropy=2.631, SmoothL1=0.848
    [Epoch 0][Batch 55], Speed: 24.741 samples/sec, CrossEntropy=2.613, SmoothL1=0.839
    [Epoch 1][Batch 0], Speed: 23.095 samples/sec, CrossEntropy=1.693, SmoothL1=0.388
    [Epoch 1][Batch 1], Speed: 21.539 samples/sec, CrossEntropy=1.636, SmoothL1=0.422
    [Epoch 1][Batch 2], Speed: 22.867 samples/sec, CrossEntropy=1.664, SmoothL1=0.473
    [Epoch 1][Batch 3], Speed: 23.164 samples/sec, CrossEntropy=1.643, SmoothL1=0.475
    [Epoch 1][Batch 4], Speed: 24.804 samples/sec, CrossEntropy=1.647, SmoothL1=0.435
    [Epoch 1][Batch 5], Speed: 22.504 samples/sec, CrossEntropy=1.657, SmoothL1=0.432
    [Epoch 1][Batch 6], Speed: 20.889 samples/sec, CrossEntropy=1.643, SmoothL1=0.442
    [Epoch 1][Batch 7], Speed: 23.014 samples/sec, CrossEntropy=1.600, SmoothL1=0.433
    [Epoch 1][Batch 8], Speed: 23.354 samples/sec, CrossEntropy=1.603, SmoothL1=0.429
    [Epoch 1][Batch 9], Speed: 25.431 samples/sec, CrossEntropy=1.606, SmoothL1=0.436
    [Epoch 1][Batch 10], Speed: 24.679 samples/sec, CrossEntropy=1.588, SmoothL1=0.435
    [Epoch 1][Batch 11], Speed: 19.304 samples/sec, CrossEntropy=1.584, SmoothL1=0.452
    [Epoch 1][Batch 12], Speed: 24.501 samples/sec, CrossEntropy=1.563, SmoothL1=0.450
    [Epoch 1][Batch 13], Speed: 24.346 samples/sec, CrossEntropy=1.554, SmoothL1=0.463
    [Epoch 1][Batch 14], Speed: 22.783 samples/sec, CrossEntropy=1.577, SmoothL1=0.478
    [Epoch 1][Batch 15], Speed: 20.915 samples/sec, CrossEntropy=1.572, SmoothL1=0.484
    [Epoch 1][Batch 16], Speed: 25.523 samples/sec, CrossEntropy=1.587, SmoothL1=0.480
    [Epoch 1][Batch 17], Speed: 22.855 samples/sec, CrossEntropy=1.590, SmoothL1=0.488
    [Epoch 1][Batch 18], Speed: 23.134 samples/sec, CrossEntropy=1.589, SmoothL1=0.487
    [Epoch 1][Batch 19], Speed: 22.281 samples/sec, CrossEntropy=1.583, SmoothL1=0.477
    [Epoch 1][Batch 20], Speed: 23.752 samples/sec, CrossEntropy=1.585, SmoothL1=0.474
    [Epoch 1][Batch 21], Speed: 25.520 samples/sec, CrossEntropy=1.571, SmoothL1=0.472
    [Epoch 1][Batch 22], Speed: 25.307 samples/sec, CrossEntropy=1.563, SmoothL1=0.468
    [Epoch 1][Batch 23], Speed: 25.239 samples/sec, CrossEntropy=1.558, SmoothL1=0.462
    [Epoch 1][Batch 24], Speed: 23.763 samples/sec, CrossEntropy=1.551, SmoothL1=0.458
    [Epoch 1][Batch 25], Speed: 26.783 samples/sec, CrossEntropy=1.544, SmoothL1=0.449
    [Epoch 1][Batch 26], Speed: 24.216 samples/sec, CrossEntropy=1.540, SmoothL1=0.451
    [Epoch 1][Batch 27], Speed: 22.234 samples/sec, CrossEntropy=1.536, SmoothL1=0.454
    [Epoch 1][Batch 28], Speed: 21.781 samples/sec, CrossEntropy=1.536, SmoothL1=0.448
    [Epoch 1][Batch 29], Speed: 25.945 samples/sec, CrossEntropy=1.534, SmoothL1=0.446
    [Epoch 1][Batch 30], Speed: 24.457 samples/sec, CrossEntropy=1.544, SmoothL1=0.446
    [Epoch 1][Batch 31], Speed: 21.982 samples/sec, CrossEntropy=1.539, SmoothL1=0.446
    [Epoch 1][Batch 32], Speed: 25.446 samples/sec, CrossEntropy=1.539, SmoothL1=0.442
    [Epoch 1][Batch 33], Speed: 21.998 samples/sec, CrossEntropy=1.539, SmoothL1=0.440
    [Epoch 1][Batch 34], Speed: 26.112 samples/sec, CrossEntropy=1.534, SmoothL1=0.436
    [Epoch 1][Batch 35], Speed: 21.811 samples/sec, CrossEntropy=1.532, SmoothL1=0.438
    [Epoch 1][Batch 36], Speed: 21.606 samples/sec, CrossEntropy=1.532, SmoothL1=0.442
    [Epoch 1][Batch 37], Speed: 24.647 samples/sec, CrossEntropy=1.528, SmoothL1=0.439
    [Epoch 1][Batch 38], Speed: 22.758 samples/sec, CrossEntropy=1.526, SmoothL1=0.435
    [Epoch 1][Batch 39], Speed: 26.268 samples/sec, CrossEntropy=1.528, SmoothL1=0.432
    [Epoch 1][Batch 40], Speed: 25.499 samples/sec, CrossEntropy=1.521, SmoothL1=0.429
    [Epoch 1][Batch 41], Speed: 23.297 samples/sec, CrossEntropy=1.520, SmoothL1=0.431
    [Epoch 1][Batch 42], Speed: 24.008 samples/sec, CrossEntropy=1.512, SmoothL1=0.428
    [Epoch 1][Batch 43], Speed: 25.531 samples/sec, CrossEntropy=1.507, SmoothL1=0.428
    [Epoch 1][Batch 44], Speed: 24.485 samples/sec, CrossEntropy=1.502, SmoothL1=0.428
    [Epoch 1][Batch 45], Speed: 24.071 samples/sec, CrossEntropy=1.498, SmoothL1=0.426
    [Epoch 1][Batch 46], Speed: 20.921 samples/sec, CrossEntropy=1.493, SmoothL1=0.424
    [Epoch 1][Batch 47], Speed: 22.816 samples/sec, CrossEntropy=1.492, SmoothL1=0.426
    [Epoch 1][Batch 48], Speed: 22.374 samples/sec, CrossEntropy=1.488, SmoothL1=0.424
    [Epoch 1][Batch 49], Speed: 25.272 samples/sec, CrossEntropy=1.483, SmoothL1=0.422
    [Epoch 1][Batch 50], Speed: 19.745 samples/sec, CrossEntropy=1.485, SmoothL1=0.424
    [Epoch 1][Batch 51], Speed: 23.355 samples/sec, CrossEntropy=1.482, SmoothL1=0.422
    [Epoch 1][Batch 52], Speed: 23.705 samples/sec, CrossEntropy=1.481, SmoothL1=0.422
    [Epoch 1][Batch 53], Speed: 19.482 samples/sec, CrossEntropy=1.486, SmoothL1=0.425
    [Epoch 1][Batch 54], Speed: 20.724 samples/sec, CrossEntropy=1.484, SmoothL1=0.428
    [Epoch 1][Batch 55], Speed: 22.942 samples/sec, CrossEntropy=1.479, SmoothL1=0.425


Finished!

We have a test image with multiple Pikachus. Let's see if we can detect all of them.


```python
test_url = 'https://raw.githubusercontent.com/zackchase/mxnet-the-straight-dope/master/img/pikachu.jpg'
download(test_url, 'pikachu_test.jpg')
net = gluoncv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
net.load_parameters('ssd_512_mobilenet1.0_pikachu.params')
x, image = gluoncv.data.transforms.presets.ssd.load_test('pikachu_test.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()
```


![png](http://hangzh.com/ICCV19-GluonCV/images/ObjectDetection_files/ObjectDetection_36_0.png)

