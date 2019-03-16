import os
import time
import warnings
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from subprocess import call
warnings.filterwarnings("error")

def mount_blob_storage(container, path, temp_path):
    cmds = ["blobfuse", "{}", "--container-name={}", "--tmp-path={}"]
    cmds[1] = cmds[1].format(path)
    cmds[2] = cmds[2].format(container)
    cmds[3] = cmds[3].format(temp_path)
    call(cmds)
    return path

def process_image(path, label, image_size):
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_jpeg(img_raw)
    img_final = tf.image.resize(img_tensor, [image_size, image_size]) / 255
    return img_final, label

def walk_images(base_path, image_size=160):
    images = []
    print('Scanning {}'.format(base_path))
    # find subdirectories in base path
    # (they should be the labels)
    labels = []
    for (_, dirs, _) in os.walk(base_path):
        print('Found {}'.format(dirs))
        labels = dirs
        break

    for d in labels:
        path = os.path.join(base_path, d)
        # only care about files in directory
        for item in os.listdir(path):
            if not item.endswith('.jpg'):
                print('skipping {}'.format(item))
                continue

            image = os.path.join(path, item)
            try:
                img, _ = process_image(image, 0, image_size)
                assert img.shape[2] == 3, "Invalid channel count"
                # write out good images
                images.append(image)
            except Exception as e:
                print('{}\n{}\n'.format(e, image))
            except RuntimeWarning as w:
                print('--------------------------{}\n{}\n'.format(w, image))

    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-t', '--target', help='target file to hold good data', default='dataset.txt')
    parser.add_argument('-i', '--img_size', help='target image size to verify', default=160, type=int)
    args = parser.parse_args()

    # ENV set, we are mounting blob storage
    if 'BASE_PATH' in os.environ:
        base_path = mount_blob_storage(os.environ['AZURE_STORAGE_CONTAINER'], 
                                        os.environ['BASE_PATH'], 
                                        os.environ['TEMP_PATH'])
    else:
        base_path = '..'
        
    base_path = Path(base_path).joinpath(args.data).resolve()
    target_path = Path(base_path).resolve().joinpath(args.target)

    images = walk_images(str(base_path), args.img_size)

    # save file
    print('writing dataset to {}'.format(target_path))
    with open(str(target_path), 'w+') as f:
        f.write('\n'.join(images))

    # python data.py -d data/PetImages -t dataset.txt
