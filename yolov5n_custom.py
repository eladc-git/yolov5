import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, initializers, Sequential, activations

def autopad(k, pad=None):  # kernel, padding
    # Pad to 'same'
    if pad is None:
        pad = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return pad

class Pad(layers.Layer):
    def __init__(self, pad):
        super(Pad, self).__init__()
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)

    def get_config(self):
        config = super().get_config()
        config.update({"pad": self.pad})
        return config

class ConvBlock(layers.Layer):
    # Standard convolution
    def __init__(self, c_in, c_out, k=1, s=1, pad=None, bn=None, act=None):
        super(ConvBlock, self).__init__()
        self.c_in, self.c_out, self.k, self.s = c_in, c_out, k, s
        self.conv = layers.Conv2D(c_out, k, s, 'SAME' if s == 1 else 'VALID', use_bias=False if bn is not None else True)
        self.pad = 0
        if s > 1:
            self.pad = autopad(k, pad)
            self.conv = Sequential([Pad(self.pad), self.conv])
        self.bn = layers.BatchNormalization() if bn is not None else tf.identity
        # YOLOv5 activations
        if act == "leaky_relu":
            self.act = (lambda x: activations.relu(x, alpha=0.1))
            self.act_fun = "leaky_relu"
        elif act == "hard_swish":
            self.act = (lambda x: x * tf.nn.relu6(x + 3) * 0.166666667)
            self.act_fun = "hard_swish"
        elif act == "swish":
            self.act = (lambda x: activations.swish(x))
            self.act_fun = "swish"
        elif act == "sigmoid":
            self.act = (lambda x: activations.sigmoid(x))
            self.act_fun = "sigmoid"
        else:
            self.act = tf.identity
            self.act_fun = "None"

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels_in": self.c_in, "channels_out": self.c_out, "kernel": self.k,
            "stride": self.s, "pad": self.pad,
            "BN": False if self.bn==tf.identity else True,
            "Activation": self.act_fun,
        })
        return config

class BottleneckBlock(layers.Layer):
    # Standard bottleneck
    def __init__(self, c_in, c_out, shortcut=True, act=None, e=0.5):
        super(BottleneckBlock, self).__init__()
        c_hidden = int(c_out * e)  # hidden channels
        self.cv1 = ConvBlock(c_in, c_hidden, k=1, s=1, act=act)
        self.cv2 = ConvBlock(c_hidden, c_out, k=3, s=1, pad=1, act=act)
        self.add = shortcut and c_in == c_out

    def call(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({
            "Conv1": self.cv1, "Conv2": self.cv2, "AddInput": self.add,
        })
        return config

class C3Block(layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c_in, c_out, n=1, shortcut=True, act=None, e=0.5):
        super(C3Block, self).__init__()
        c_hidden = int(c_out * e)  # hidden channels
        self.cv1 = ConvBlock(c_in, c_hidden, 1, 1, act=act)
        self.cv2 = ConvBlock(c_in, c_hidden, 1, 1, act=act)
        self.m = Sequential([BottleneckBlock(c_hidden, c_hidden, shortcut, act=act, e=1.0) for _ in range(n)])
        self.cv3 = ConvBlock(2 * c_hidden, c_out, 1, 1, act=act)

    def call(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))

    def get_config(self):
        config = super().get_config()
        config.update({
            "Conv1": self.cv1, "Conv2": self.cv2, "Conv3": self.cv3, "Bottleneck": self.m,
        })
        return config

class SPPFBlock(layers.Layer):
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c_in, c_out, k=5):
        super(SPPFBlock, self).__init__()
        c_hidden = c_in // 2  # hidden channels
        self.cv1 = ConvBlock(c_in, c_hidden, 1, 1)
        self.cv2 = ConvBlock(c_hidden * 4, c_out, 1, 1)
        self.max_pooing = layers.MaxPool2D(pool_size=k, strides=1, padding='SAME')

    def call(self, inputs):
        x = self.cv1(inputs)
        y1 = self.max_pooing(x)
        y2 = self.max_pooing(y1)
        return self.cv2(tf.concat([x, y1, y2, self.max_pooing(y2)], 3))

    def get_config(self):
        config = super().get_config()
        config.update({
            "Conv1": self.cv1, "Conv2": self.cv2, "MaxPooling": self.max_pooing,
        })
        return config

class Upsample(layers.Layer):
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        assert scale_factor == 2, "scale_factor must be 2"
        self.scale_factor = scale_factor
        self.mode = mode
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), method=mode)

    def call(self, inputs):
        return self.upsample(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale_factor": self.scale_factor, "mode": self.mode,
        })
        return config

class Concat(layers.Layer):
    def __init__(self, dimension=3):
        super(Concat, self).__init__()
        self.dimension = dimension

    def call(self, inputs):
        return tf.concat(inputs, self.dimension)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dimension": self.dimension,
        })
        return config

class ConcatOut(layers.Layer):
    def __init__(self, dimension=2):
        super(ConcatOut, self).__init__()
        self.dimension = dimension
        cv1 = ConvBlock(256, 255, 1, 1, act="sigmoid")
        cv2 = ConvBlock(128, 255, 1, 1, act="sigmoid")
        cv3 = ConvBlock(64, 255, 1, 1, act="sigmoid")
        self.conv = [cv1, cv2, cv3]
        self.reshape = tf.keras.layers.Reshape((-1, 3, 85))
        self.concat = tf.keras.layers.Concatenate(axis=dimension)

    def call(self, inputs):
        y = []
        for i, x in enumerate(inputs):
            x = tf.transpose(self.reshape(self.conv[i](x)), [0, 2, 1, 3])
            y.append(x)
        return self.concat(y)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_inputs": len(self.conv), "dimension": self.dimension,
            "Conv1": self.conv[0], "Conv2": self.conv[1], "Conv3": self.conv[2],
        })
        return config

class Yolov5n:
    def __init__(self, input_shape, batch_size=1):
        self.input_shape = input_shape
        inputs = layers.Input(shape=input_shape, batch_size=batch_size)
        x = ConvBlock(c_in=3, c_out=16, k=6, s=2, pad=2, act="swish")(inputs)
        x = ConvBlock(c_in=16, c_out=32, k=3, s=2, pad=1, act="swish")(x)
        x = C3Block(c_in=32, c_out=32, n=1, act="swish")(x)
        x = ConvBlock(c_in=32, c_out=64, k=3, s=2, pad=1, act="swish")(x)
        x = C3Block(c_in=64, c_out=64, n=2, act="swish")(x)
        x1 = ConvBlock(c_in=64, c_out=128, k=3, s=2, pad=1, act="swish")(x)
        x1 = C3Block(c_in=128, c_out=128, n=3, act="swish")(x1)
        x2 = ConvBlock(c_in=128, c_out=256, k=3, s=2, pad=1, act="swish")(x1)
        x2 = C3Block(c_in=256, c_out=256, n=1, act="swish")(x2)
        x2 = SPPFBlock(c_in=256, c_out=256)(x2)
        x2 = ConvBlock(c_in=256, c_out=128, k=1, s=1, act="swish")(x2)
        x3 = Upsample(scale_factor=2, mode="nearest")(x2)
        x3 = Concat()([x1,x3])
        x3 = C3Block(c_in=256, c_out=128, n=1, act="swish")(x3)
        x3 = ConvBlock(c_in=128, c_out=64, k=1, s=1, pad=1, act="swish")(x3)
        x4 = Upsample(scale_factor=2, mode="nearest")(x3)
        x4 = Concat()([x,x4])
        y1 = C3Block(c_in=128, c_out=64, n=1, act="swish")(x4)
        x4 = ConvBlock(c_in=64, c_out=64, k=3, s=2, pad=1, act="swish")(y1)
        x4 = Concat()([x3,x4])
        y2 = C3Block(c_in=64, c_out=128, n=1, act="swish")(x4)
        x4 = ConvBlock(c_in=128, c_out=128, k=3, s=2, pad=1, act="swish")(y2)
        x4 = Concat()([x2,x4])
        y3 = C3Block(c_in=256, c_out=256, n=1, act="swish")(x4)
        y = ConcatOut()([y1,y2,y3])
        self.model = Model(inputs=inputs, outputs=y, name="yolov5n")
        self.model.trainable = False
        self.model.summary()

    def get_model(self):
        return self.model

    def get_input_shape(self):
        return self.input_shape

    def load_weights(self, weight_path):
        weights = np.load(weight_path, allow_pickle=True)
        self.model.set_weights(weights)

def create_yolov5n(image_size, batch_size=1, model_path=None):
    yolov5n = Yolov5n(input_shape=image_size, batch_size=batch_size)
    if model_path is not None:
        yolov5n.load_weights(model_path)
    model = yolov5n.get_model()
    return model

if __name__ == "__main__":

    image_size = (640,640,3)
    model = create_yolov5n(image_size, model_path=r"E:\Models\yolov5\yolov5n_weights.npy")

    np.save("yolov5n_weights", model.get_weights())
    exit()

    # import pickle
    # names = [weight.name for layer in model.layers for weight in layer.weights]
    # weights = model.get_weights()
    # weights_dict = {}
    # for name, weight in zip(names, weights):
    #     weights_dict[name] = weight
    # a_file = open("data.pkl", "wb")
    # pickle.dump(weights_dict, a_file)
    # a_file.close()

    y = model(tf.random.uniform((1,*image_size)))
    model.save("yolov5n.h5", save_format='h5')
    print("Done! Model is saved!")

    # m = tf.keras.models.load_model("yolov5n.h5")
    # print("done!")
    # exit()