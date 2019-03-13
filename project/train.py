from __future__ import absolute_import, division, print_function
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from random import shuffle
from tensorflow.data import Dataset
import warnings
warnings.filterwarnings("error")

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_SIZE = 160 # All images will be resized to 160x160
BATCH_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_path = 'data/PetImages'

def load_files(base_path, dataset=None, split=[8, 1, 1]):
    # normalize splits
    splits = np.array(split) / np.sum(np.array(split))
    
    # find labels - parent folder names
    labels = { k.name: v for (v, k) in enumerate(Path(base_path).glob('*/')) }
    
    # load all files along with idx label
    if dataset != None:
        with open(dataset, 'r') as d:
            data = [(str(Path(f.strip()).absolute()), labels[Path(f.strip()).parent.name]) for f in d.readlines()]
    else:
        data = [(str(f.absolute()), labels[f.parent.name]) for f in Path(base_path).glob('**/*.jpg')]

    print(data[len(data)-1])
    
    # shuffle data
    shuffle(data)
    
    # split data
    train_idx = int(len(data) * splits[0])
    eval_idx = int(len(data) * splits[1])
    
    return data[:train_idx], \
            data[train_idx:train_idx + eval_idx], \
            data[train_idx + eval_idx:], \
            labels

def get_dataset(base_path, file):
    warnings.filterwarnings('error')
    data = [str(f.absolute()) for f in Path(base_path).glob('**/*.jpg')]
    bad = []
    with open(file, "w") as f:
        for item in data:
            try:
                img, _ = process_item(item, 0)
                assert img.shape[2] == 3, "Invalid channel count"
                # write out good images
                f.write('{}\n'.format(item))
            except Exception as e:
                bad.append(item)
                print('{}\n{}\n'.format(e, item))
            except RuntimeWarning as w:
                bad.append(item)
                print('--------------------------{}\n{}\n'.format(w, item))

        print('{} bad images ({})'.format(len(bad), len(bad) / float(len(data))))
    return bad


def process_item(path, label):
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_jpeg(img_raw)
    img_final = tf.image.resize(img_tensor, [IMG_SIZE, IMG_SIZE]) / 255
    return img_final, label

def run():
    train, test, val, labels = load_files(base_path, dataset='dataset.txt')

    # training data
    train_data, train_labels = zip(*train)
    train_ds = Dataset.zip((Dataset.from_tensor_slices(list(train_data)),
                            Dataset.from_tensor_slices(list(train_labels))))

    train_ds = train_ds.map(map_func=process_item, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.repeat()

    # model
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1)
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

    history = model.fit(train_ds, epochs=1, steps_per_epoch=tf.math.ceil(len(train)/BATCH_SIZE).numpy())
    model.save('./model/trained.h5')
    tf.keras.experimental.export_saved_model(model, './model/saved_model.h5')

def load(path):
    #new_model = tf.keras.models.load_model(path)
    new_model = tf.keras.experimental.load_from_saved_model('./model/saved_model.h5')
    img = "C:\\projects\\k8s-ml\\project\\data\\PetImages\\Dog\\10020.jpg"
    tensor, label = process_item(img, 0)
    t = tf.reshape(tensor,[-1, *IMG_SHAPE])
    print(t.shape)
    o = new_model.predict(t)
    print(o)
    print('Done!')

if __name__ == "__main__":
    get_dataset(base_path, 'dataset.txt')
    #run()
    #load('./model/trained.h5')
    #load_files(base_path)
    #load_files(base_path, 'dataset.txt')
    #get_dataset(base_path, 'dataset.txt')