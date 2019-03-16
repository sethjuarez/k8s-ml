import json
import time
import datetime
import numpy as np
from helpers import info, process_image
import tensorflow as tf
from io import StringIO

#from azureml.core.model import Model

def load(path):
    #new_model = tf.keras.models.load_model(path)
    #new_model = tf.keras.experimental.load_from_saved_model('./model/saved_model.h5')

    model = tf.keras.models.load_model('model/model.h5')
    model.summary()
    img = "C:\\projects\\k8s-ml\\project\\data\\PetImages\\Dog\\10020.jpg"
    print('Predict {}'.format(img))
    tensor, label = process_image(img, 0, 160)
    t = tf.reshape(tensor,[-1, 160, 160, 3])
    o = model.predict(t)
    print(o)

    img = "C:\\projects\\k8s-ml\\project\\data\\PetImages\\Cat\\2.jpg"
    print('Predict {}'.format(img))
    tensor, label = process_image(img, 0, 160)
    t = tf.reshape(tensor,[-1, 160, 160, 3])
    o = model.predict(t)
    print(o)

    print('Done!')

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


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def run(raw_data):
    global model
    prev_time = time.time()
          
    post = json.loads(raw_data)
    

    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    payload = {
        'time': inference_time.total_seconds(),
        'prediction': 3,
        'scores': [1,2,2]
    }

    print('Input ({}), Prediction ({})'.format(post['image'], payload))

    return payload

if __name__ == "__main__":
    info('Using TensorFlow v.{}'.format(tf.__version__))
    g = load("./model")
    #init()
    #out = run({x="@"})