import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras import layers, Model, initializers, Sequential, activations
from tensorflow.keras import backend as K


class ModelCompare:
    def __init__(self, model1_path, model2_path):
        self.model1 = tf.keras.models.load_model(model1_path)
        self.model2 = tf.keras.models.load_model(model2_path)

    def get_layers_output(self, model, input_tensor):
        intermediate_model = Model(inputs=model.layers[0].input, outputs=[l.output for l in model.layers[1:]])
        y = intermediate_model.predict(input_tensor)
        return y

    def compare(self, input_tensor):
        y1 = self.get_layers_output(self.model1, input_tensor)
        y2 = self.get_layers_output(self.model2, input_tensor)
        return y1, y2

if __name__ == "__main__":
    image_size = (640, 640, 3)
    model1_path = r"E:\Models\yolov5\yolov5n.h5"
    model2_path = r"E:\Models\yolov5\yolov5n_quantized.h5"
    input = tf.random.uniform((1, *image_size))
    modelCompare = ModelCompare(model1_path, model2_path)
    y1, y2 = modelCompare.compare(input)
    print(y1)


