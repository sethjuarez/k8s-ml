import os
import argparse
import onnxmltools
import tensorflow as tf
from pathlib import Path
from subprocess import call

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def mount_blob_storage(container, path, temp_path):
    cmds = ["blobfuse", "{}", "--container-name={}", "--tmp-path={}"]
    cmds[1] = cmds[1].format(path)
    cmds[2] = cmds[2].format(container)
    cmds[3] = cmds[3].format(temp_path)
    call(cmds)
    return path

def run(model_file, target_file):
    info('Attempting conversion')
    print('Converting:\n{} to\n{}\n'.format(model_file, target_file))
    print('Attempting to load model')
    model = tf.keras.models.load_model(model_file)

    # reshaping input to concrete input
    print('Reshaping input layer to fixed unit batch size')
    fixed = tf.keras.Input(shape=(160, 160, 3), 
                                       name="x", 
                                       dtype=tf.float32, 
                                       batch_size=1)

    new_model = tf.keras.Sequential([fixed, model])

    new_model.summary()
    print('Done!\nConversion')    
    onnx_model = onnxmltools.convert_keras(new_model) 
    onnxmltools.utils.save_model(onnx_model, target_file)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert keras model to onnx')
    parser.add_argument('-m', '--model', help='relative model file')
    parser.add_argument('-t', '--target', help='target onnx file')
    args = parser.parse_args()

    print('Using TensorFlow v.{}'.format(tf.__version__))
    # ENV set, we are mounting blob storage
    if 'BASE_PATH' in os.environ:
        base_path = mount_blob_storage(os.environ['AZURE_STORAGE_CONTAINER'], 
                                        os.environ['BASE_PATH'], 
                                        os.environ['TEMP_PATH'])
    else:
        base_path = '..'
        
    source_path = Path(base_path).joinpath(args.model).resolve()
    target_path = Path(base_path).resolve().joinpath(args.target)

    run(str(source_path), str(target_path))

    # python convert.py -m model/latest_model.h5 -t model/latest.onnx
    # blobfuse $BASE_PATH --container-name=$AZURE_STORAGE_CONTAINER --tmp-path=$TEMP_PATH