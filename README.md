ICCV 2019 Tutorial: Everything You Need to Know to Reproduce SOTA Deep Learning Models
======================================================================================

<h3>Time: Sunday, October 27, 2019. Half Day - AM (0800-1215)<br/>Location: Auditorium, [COEX Convention Center](https://goo.gl/maps/VKDgLyYf8NDC1e4E6)</h3>

<span style="color:grey">Presenter: Zhi Zhang, Sam Skalicky, Muhyun Kim, Jiyang Kang</span><br/>



<a href="https://aws.amazon.com/"><img src="_static/aws_logo.png" alt="AWS Icon" height="45"></a> &nbsp; <a href="https://aws.amazon.com/"><img src="_static/amazon_ai.png" alt="AmazonAI Icon" height="58"></a> &nbsp; <a href="https://aws.amazon.com/sagemaker/neo/"><img src="_static/neo.png" alt="Neo Icon" height="58"></a> &nbsp; <a href="https://http://mxnet.incubator.apache.org/"><img src="_static/apache_incubator_logo.png" alt="Apache Incubator Icon" height="39"></a> &nbsp; <a href="https://http://mxnet.incubator.apache.org/"><img src="_static/mxnet_logo_2.png" alt="MXNet Icon" height="39"></a> &nbsp; <a href="https://gluon-cv.mxnet.io/"><img src="_static/gluon_logo_horizontal_small.png" alt="Gluon Icon" height="42"></a> &nbsp; <a href="http://tvm.ai"><img src="_static/tvm.png" alt="TVM Icon" height="32"></a>

Abstract
--------

Deep Learning has become the de facto standard algorithm in computer vision. There are a surge amount of approaches being proposed every year for different tasks. Reproducing the complete system in every single detail can be problematic and time-consuming, especially for the beginners. Existing open-source implementations are typically not well-maintained and the code can be easily broken by the rapid updates of the deep learning frameworks. In this tutorial, we will walk through the technical details of the state-of-the-art (SOTA) algorithms in major computer vision tasks, and we also provide the code implementations and hands-on tutorials to reproduce the large-scale training in this tutorial.

Agenda
------

| Time        | Title                                                                  | Slides    | Notebooks  |
|-------------|------------------------------------------------------------------------|-----------|------------|
| 8:00-8:15   | Welcome and AWS Setup(Free instance available)                         |           | [link][01] |
| 8:15-8:40   | Introduction to MXNet and GluonCV                                      | [link][1],[link][0] |            |
| 8:40-9:00   | Deep Learning and Gluon Basics (NDArray, AutoGrad, Libraries)          |           | [link][11],[link][12] |
| 9:00-9:30   | Bag of Tricks for Image Classification (ResNet, MobileNet, Inception)  | [link][2] | [link][21] |
| 9:30-10:00  | Bag of Freebies for Object Detectors (SSD, Faster RCNN, YOLOV3)        | [link][3] | [link][31] |
| 10:00-10:30 | Semantic segmentation algorithms (FCN, PSPNet, DeepLabV3, VPLR)        | [link][4] | [link][41] |
| 10:30-11:00 | Pose Estimation(SimplePose, AlphaPose)                                 | [link][6] | [link][61] |
| 11:00-11:30 | Action Recognition(TSN, I3D)                                           | [link][7] |            |
| 11:30-12:00 | Painless Deployment (C++, TVM)                                         | [link][5] | [link][51],[link][52] |
| 12:00-12:15 | Q&A and Closing                                                        |           |            |

Q&A
---

Q1: How do I setup the environments for this tutorial?

A1: There will be all-in-one AWS SageMaker notebooks available for all local attendees, you need to bring your laptop and have a working email to access the notebooks.


[0]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/slides/IntroToGluonCV.pptx
[1]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/slides/MXNet_Overview.pptx
[2]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/slides/ImageClassification.pptx
[3]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/slides/ObjectDetection.pptx
[4]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/slides/Segmentation.pptx
[5]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/slides/Deployment.pptx
[6]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/slides/PoseEstimation.pptx
[7]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/slides/ActionRecognition.pptx

[01]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/00_setup/use_aws.ipynb
[11]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/01_basics/autograd.ipynb
[12]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/01_basics/ndarray.ipynb
[21]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/02_classification/ImageClassification.ipynb
[31]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/03_detection/ObjectDetection.ipynb
[41]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/04_segmentation/SemanticSegmentation.ipynb
[51]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/07_deployment/export_network.ipynb
[52]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/07_deployment/cpp_inference.ipynb
[61]: https://github.com/zhreshold/ICCV19-GluonCV/blob/master/05_pose/PoseEstimation.ipynb


Organizers
---

<span style="color:grey">Hang Zhang, Tong He, Zhi Zhang, Zhongyue Zhang, Haibin Lin, Aston Zhang, Mu Li</span>
