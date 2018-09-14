Chainer Segmentation Handson
============================

# Extract images from tfrecord files by yourself

If you want to extract images from `.tfrecord` files provided by NVIDIA by yourself, please download them first.

```
curl -L -O https://github.com/mitmul/DeepLearningInstitude-Handson/releases/download/SegmentationDataset/train_images.tfrecords
curl -L -O https://github.com/mitmul/DeepLearningInstitude-Handson/releases/download/SegmentationDataset/val_images.tfrecords
```

Then, please run this script to save all images included in the `.tfrecord` files as PNG files:

```
python tfrecord_extractor.py
```

Note that you need to install OpenCV and TensorFlow before running the script above.

```
conda install -c menpo opencv3
pip install tensorflow
```
