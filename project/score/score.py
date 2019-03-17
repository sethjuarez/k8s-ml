import json
import time
import datetime
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

#from azureml.core.model import Model

global model

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

def process_image(path, image_size):
    # Extract image (from web or path)
    if(path.startswith('http')):
        response = requests.get(path)
        img = np.array(Image.open(BytesIO(response.content)))
    else:
        img = np.array(Image.open(path))

    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_final = tf.image.resize(img_tensor, [image_size, image_size]) / 255
    return img_final

def load(path):
    global model
    
    info('Loading Model')
    try:
        print('Attempting AzureML model load from {}...'.format(path))
        model_path = Model.get_model_path(path)
        print('Successfully found AzureML Model')
    except:
        print('...failed, falling back to local {}'.format(path))
        model_path = path

    print('Attempting to load model')
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print('Done!')

    #img = "C:\\projects\\k8s-ml\\project\\data\\PetImages\\Dog\\10020.jpg"
    #print('Predict {}'.format(img))
    #tensor, label = process_image(img, 0, 160)
    #t = tf.reshape(tensor,[-1, 160, 160, 3])
    #o = model.predict(t)
    #print(o)

    #img = "C:\\projects\\k8s-ml\\project\\data\\PetImages\\Cat\\2.jpg"
    #print('Predict {}'.format(img))
    #tensor, label = process_image(img, 0, 160)
    #t = tf.reshape(tensor,[-1, 160, 160, 3])
    #o = model.predict(t)
    #print(o)

    #print('Done!')

    # load code.....
    #model = tf.keras.models.load_model('model/model.h5')
    #model = tf.saved_model.load(path)
    #print('Done!')
    #model.summary()
    return model

def init():
    global model

    try:
        model_path = Model.get_model_path('modelname')
    except:
        model_path = 'model.pb'

    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='model')

    print('Initialized model "{}" at {}'.format(model_path, datetime.datetime.now()))
    model = graph

def run(raw_data):
    global model
    info('Inference')
    prev_time = time.time()
          
    post = json.loads(raw_data)
    img_path =  post['image']
    

    current_time = time.time()

    tensor = process_image(img_path, 160)
    t = tf.reshape(tensor,[-1, 160, 160, 3])
    o = model.predict(t)[0][0]
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    payload = {
        'time': inference_time.total_seconds(),
        'prediction': 'Cat' if o < 0 else 'Dog',
        'scores': o
    }

    print('Input ({}), Prediction ({})'.format(post['image'], payload))

    return payload

if __name__ == "__main__":
    info('Using TensorFlow v.{}'.format(tf.__version__))

    cat = 'https://images.unsplash.com/photo-1518791841217-8f162f1e1131'
    dog = 'https://images.pexels.com/photos/356378/pexels-photo-356378.jpeg'

    x = process_image(cat, 160)
    model = load('../model/latest_model.h5')
    print('Cat:')
    run(json.dumps({ 'image': cat }))
    print('Dog:')
    run(json.dumps({ 'image': dog }))
    


#    g = load("./model")
    #init()
    #out = run({x="@"})