
import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from random import shuffle
from functools import wraps
from inspect import getargspec

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

def process_image(path, label, image_size):
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_jpeg(img_raw)
    img_final = tf.image.resize(img_tensor, [image_size, image_size]) / 255
    return img_final, label

def print_info(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        info('-> {}'.format(f.__name__))
        print('Parameters:')
        ps = list(zip(getargspec(f).args, args))
        width = max(len(x[0]) for x in ps) + 1
        for t in ps:
            items = str(t[1]).split('\n')
            print('   {0:<{w}} ->  {1}'.format(t[0], items[0], w=width))
            for i in range(len(items) - 1):
                print('   {0:<{w}}       {1}'.format(' ', items[i+1], w=width))
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print('\n -- Elapsed {0:.4f}s\n'.format(te-ts))
        return result
    return wrapper
    
def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def check_dir(path, check=False):
    if check:
        assert os.path.exists(path), '{} does not exist!'.format(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path).resolve()
