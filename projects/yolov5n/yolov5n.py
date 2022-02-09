import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras import layers, Model, initializers, Sequential, activations
from tensorflow.keras import backend as K

def autopad(k, pad=None):  # kernel, padding
    # Pad to 'same'
    if pad is None:
        pad = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return pad


def ConvBlock(x, c_in, c_out, k=1, s=1, pad=None, bn=None, act=None):
    # Standard convolution
    if act == "leaky_relu":
        act_fun = lambda x: activations.relu(x, alpha=0.1)
    elif act == "hard_swish":
        act_fun = lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif act == "swish":
        act_fun = lambda x: activations.swish(x)
    elif act == "sigmoid":
        act_fun = lambda x: activations.sigmoid(x)
    else:
        act_fun = tf.identity

    if s > 1:
        pad = autopad(k, pad)
        constant = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        x = tf.pad(x, constant, mode='constant', constant_values=0)
    x = layers.Conv2D(c_out, k, s, 'SAME' if s == 1 else 'VALID', use_bias=False if bn is not None else True)(x)
    if bn is not None:
        x = layers.BatchNormalization()(x)
    x = act_fun(x)
    return x


def BottleneckBlock(x, c_in, c_out, shortcut=True, act=None, e=0.5):
    # Standard bottleneck
    c_hidden = int(c_out * e)  # hidden channels
    y = ConvBlock(x, c_in, c_hidden, k=1, s=1, act=act)
    y = ConvBlock(y, c_hidden, c_out, k=3, s=1, pad=1, act=act)
    add = shortcut and c_in == c_out
    if add:
        y += x
    return y


def C3Block(x, c_in, c_out, n=1, shortcut=True, act=None, e=0.5):
    # CSP Bottleneck with 3 convolutions
    c_hidden = int(c_out * e)  # hidden channels
    y = ConvBlock(x, c_in, c_hidden, k=1, s=1, act=act)
    z = ConvBlock(x, c_in, c_hidden, k=1, s=1, act=act)
    for _ in range(n):
        y = BottleneckBlock(y, c_hidden, c_hidden, shortcut, act=act, e=1.0)
    y = tf.concat((y, z), axis=3)
    y = ConvBlock(y, 2 * c_hidden, c_out, k=1, s=1, act=act)
    return y


def SPPFBlock(x, c_in, c_out, k=5):
    # Spatial pyramid pooling-Fast layer
    c_hidden = c_in // 2  # hidden channels
    x = ConvBlock(x, c_in, c_hidden, 1, 1, act="swish")
    y = layers.MaxPool2D(pool_size=k, strides=1, padding='SAME')(x)
    z = layers.MaxPool2D(pool_size=k, strides=1, padding='SAME')(y)
    w = layers.MaxPool2D(pool_size=k, strides=1, padding='SAME')(z)
    x = tf.concat([x, y, z, w], axis=3)
    x = ConvBlock(x, c_hidden * 4, c_out, 1, 1, act="swish")
    return x


def Upsample(x, scale_factor=2, mode="nearest"):
    assert scale_factor == 2, "scale_factor must be 2"
    return tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), method=mode)


def Concat(x):
    return tf.concat(x, axis=3)


def OutputLayer(x):
    assert len(x) == 3, "input length must be 3"
    x1 = ConvBlock(x[0], 256, 255, 1, 1, act="sigmoid")
    x1 = layers.Reshape((-1, 3, 85))(x1)
    x1 = tf.transpose(x1, [0, 2, 1, 3])

    x2 = ConvBlock(x[1], 128, 255, 1, 1, act="sigmoid")
    x2 = layers.Reshape((-1, 3, 85))(x2)
    x2 = tf.transpose(x2, [0, 2, 1, 3])

    x3 = ConvBlock(x[2], 64, 255, 1, 1, act="sigmoid")
    x3 = layers.Reshape((-1, 3, 85))(x3)
    x3 = tf.transpose(x3, [0, 2, 1, 3])

    return [x1,x2,x3]


class Yolov5n:
    def __init__(self, input_shape, batch_size=1):
        self.input_shape = input_shape
        inputs = layers.Input(shape=input_shape, batch_size=batch_size)
        x = ConvBlock(inputs, c_in=3, c_out=16, k=6, s=2, pad=2, act="swish")
        x = ConvBlock(x, c_in=16, c_out=32, k=3, s=2, pad=1, act="swish")
        x = C3Block(x, c_in=32, c_out=32, n=1, act="swish")
        x = ConvBlock(x, c_in=32, c_out=64, k=3, s=2, pad=1, act="swish")
        x = C3Block(x, c_in=64, c_out=64, n=2, act="swish")
        x1 = ConvBlock(x, c_in=64, c_out=128, k=3, s=2, pad=1, act="swish")
        x1 = C3Block(x1, c_in=128, c_out=128, n=3, act="swish")
        x2 = ConvBlock(x1, c_in=128, c_out=256, k=3, s=2, pad=1, act="swish")
        x2 = C3Block(x2, c_in=256, c_out=256, n=1, act="swish")
        x2 = SPPFBlock(x2, c_in=256, c_out=256)
        x2 = ConvBlock(x2, c_in=256, c_out=128, k=1, s=1, act="swish")
        x3 = Upsample(x2, scale_factor=2, mode="bilinear")
        x3 = Concat([x3, x1])
        x3 = C3Block(x3, c_in=256, c_out=128, n=1, shortcut=False, act="swish")
        x3 = ConvBlock(x3, c_in=128, c_out=64, k=1, s=1, pad=1, act="swish")
        x4 = Upsample(x3, scale_factor=2, mode="bilinear")
        x4 = Concat([x4, x])
        y1 = C3Block(x4, c_in=128, c_out=64, n=1, shortcut=False, act="swish")
        x4 = ConvBlock(y1, c_in=64, c_out=64, k=3, s=2, pad=1, act="swish")
        x4 = Concat([x4, x3])
        y2 = C3Block(x4, c_in=64, c_out=128, n=1, shortcut=False, act="swish")
        x4 = ConvBlock(y2, c_in=128, c_out=128, k=3, s=2, pad=1, act="swish")
        x4 = Concat([x4, x2])
        y3 = C3Block(x4, c_in=256, c_out=256, n=1, shortcut=False, act="swish")
        y = OutputLayer([y1, y2, y3])
        self.model = Model(inputs=inputs, outputs=y, name="yolov5n")
        self.model.trainable = False
        self.model.summary()

    def get_model(self):
        return self.model

    def get_input_shape(self):
        return self.input_shape

    def find_in_dictionary(self, dictionary, name):
        for key in dictionary.keys():
            if key.find(name) != -1:
                return key
        return None

    def print_weights_names(self):
        names = [weight.name for layer in self.model.layers for weight in layer.weights]
        for name, weight in zip(names, self.model.get_weights()):
            print(name, weight.shape)

    def load_weights(self, weights_pickle_path):
        file = open(weights_pickle_path, "rb")
        weights_dict = pickle.load(file)
        assert isinstance(weights_dict, dict), "Weights should be dictionary!"
        counter = 0
        try:
            for layer in self.model.layers:
                new_weights = []
                for weight in layer.weights:
                    key = self.find_in_dictionary(weights_dict, weight.name)
                    if key is not None:
                        new_weights.append(weights_dict[key])
                        counter += 1
                layer.set_weights(new_weights)
        except:
            assert False, "No matching in weights shapes!"
        assert len(weights_dict) == counter, "Number of weights should be equal!"

    def save_weights(self):
        import pickle
        names = [weight.name for layer in self.model.layers for weight in layer.weights]
        weights = self.model.get_weights()
        weights_dict = {}
        for name, weight in zip(names, weights):
            weights_dict[name] = weight
        file = open("weights.pkl", "wb")
        pickle.dump(weights_dict, file)
        file.close()

    def get_layers_output(self, input_tensor):
        intermediate_model = Model(inputs=self.model.layers[0].input, outputs=[l.output for l in self.model.layers[1:]])
        y = intermediate_model.predict(input_tensor)
        return y


def create_yolov5n(image_size, batch_size=1, weights_pickle_path=None):
    yolov5n = Yolov5n(input_shape=image_size, batch_size=batch_size)
    if weights_pickle_path is not None:
        yolov5n.load_weights(weights_pickle_path)
    return yolov5n


if __name__ == "__main__":
    image_size = (640, 640, 3)
    yolov5n = create_yolov5n(image_size, weights_pickle_path=r"E:\Models\yolov5\yolov5n_weights.pkl")
    input = tf.random.uniform((1, *image_size))
    y = yolov5n.model(input)
    x = yolov5n.get_layers_output(input)

    #yolov5n.model.save("yolov5n.h5", save_format='h5')
    print("Model is saved!")
