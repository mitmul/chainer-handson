#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf

import cv2 as cv

for filename, out_prefix, n_imgs in [
        ('val_images.tfrecords', 'val', 26),
        ('train_images.tfrecords', 'train', 234)]:
    if not os.path.exists('{}/image'.format(out_prefix)):
        os.makedirs('{}/image'.format(out_prefix))
    if not os.path.exists('{}/label'.format(out_prefix)):
        os.makedirs('{}/label'.format(out_prefix))
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([filename])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # list the features we want to extract, i.e., the image and the label
        features = tf.parse_single_example(
            serialized_example,
            features={
                'img_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string),
            })

        # Decode the training image
        # Convert from a scalar string tensor (whose single string has
        # length 256*256) to a float tensor
        image = tf.decode_raw(features['img_raw'], tf.int64)
        image.set_shape([65536])
        image_re = tf.reshape(image, (256, 256))

        # decode the label image, an image with all 0's except 1's where the left
        # ventricle exists
        label = tf.decode_raw(features['label_raw'], tf.uint8)
        label.set_shape([65536])
        label_re = tf.reshape(label, [256, 256])

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(n_imgs):
            image, label = sess.run([image_re, label_re])
            cv.imwrite('{}/image/{:03d}.png'.format(out_prefix, i), image)
            cv.imwrite('{}/label/{:03d}.png'.format(out_prefix, i), label)
        coord.request_stop()
        coord.join(threads)
