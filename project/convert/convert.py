import onnxmltools
import tensorflow as tf

def run():
    model = tf.saved_model.load("./model")
    print(model)

    onnx_model = onnxmltools.convert_keras(model, target_opset=7) 


if __name__ == "__main__":
    run()