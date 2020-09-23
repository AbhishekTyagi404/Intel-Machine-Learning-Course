# AI-Data-center-to-the-Edge---INTEL-AI-course
In this project, using Inception v3 model USA's most stolen cars was analysed and modeled to predict the most stolen car.

# Inception v3

The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.


The necessary python packages to install is given in environment.yml file


# DATASET

This is an overview of the VMMR dataset introduced in "A Large and Diverse Dataset for Improved Vehicle Make and Model Recognition".

Overview
Despite the ongoing research and practical interests, car make and model analysis only attracts few attentions in the computer vision community. We believe the lack of high quality datasets greatly limits the exploration of the community in this domain. To this end, we collected and organized a large-scale and comprehensive image database called VMMRdb, where each image is labeled with the corresponding make, model and production year of the vehicle.

Description
The Vehicle Make and Model Recognition dataset (VMMRdb) is large in scale and diversity, containing 9,170 classes consisting of 291,752 images, covering models manufactured between 1950 and 2016. VMMRdb dataset contains images that were taken by different users, different imaging devices, and multiple view angles, ensuring a wide range of variations to account for various scenarios that could be encountered in a real-life scenario. The cars are not well aligned, and some images contain irrelevant background. The data covers vehicles from 712 areas covering all 412 sub-domains corresponding to US metro areas. Our dataset can be used as a baseline for training a robust model in several real-life scenarios for traffic surveillance.

VMMRdb data distribution

The distribution of images in different classes of the dataset. Each circle is associated with a class, and its size represents the number of images in the class. The classes with labels are the ones including more than 100 images.


Citation
If you use this dataset, please cite the following paper:

A Large and Diverse Dataset for Improved Vehicle Make and Model Recognition
F. Tafazzoli, K. Nishiyama and H. Frigui
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops 2017. 
