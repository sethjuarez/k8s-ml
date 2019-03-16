from __future__ import absolute_import, division, print_function
import os
import math
import argparse
#import onnxmltools
import numpy as np
import tensorflow as tf
from pathlib import Path
from random import shuffle
from tensorflow.data import Dataset
from helpers import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

#@print_info
def run(base_path, image_size=160, epochs=10, batch_size=32, learning_rate=0.0001, output='model', dataset=None):
    img_shape = (image_size, image_size, 3)

    info('Creating Data Pipeline')
    # load dataset
    train, test, val, labels = load_files(base_path, dataset=dataset)

    # training data
    train_data, train_labels = zip(*train)
    train_ds = Dataset.zip((Dataset.from_tensor_slices(list(train_data)),
                            Dataset.from_tensor_slices(list(train_labels))))

    train_ds = train_ds.map(map_func=lambda p, l: process_image(p, l, image_size), 
                            num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.repeat()

    # model
    info('Creating Model')
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False, 
                                               weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

    model.summary()

    # training
    info('Training')
    steps_per_epoch = math.ceil(len(train)/batch_size)
    history = model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch)

    # save model
    info('Saving Model')
    print('Serializing model to {}'.format(output))
    #tf.saved_model.save(model, str(output))

    model.save(str(output.joinpath('model.h5')))

    #tf.keras.experimental.export_saved_model(model, str(output))

    #onnx_model = onnxmltools.convert_keras(model, target_opset=7) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=32, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.0001, type=float)
    parser.add_argument('-o', '--outputs', help='output directory', default='model')
    parser.add_argument('-f', '--dataset', help='cleaned data listing')
    args = parser.parse_args()

    info('Using TensorFlow v.{}'.format(tf.__version__))

    args.data = check_dir(args.data).resolve()
    args.outputs = check_dir(args.outputs).resolve()

    run(base_path=args.data, image_size=160, epochs=args.epochs, batch_size=args.batch, learning_rate=args.lr, output=args.outputs, dataset=args.dataset)

    #python train.py -d data/PetImages -e 1 -b 32 -l 0.0001 -o model -f dataset.txt