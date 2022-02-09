import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=64):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


class ModelFxpCompare:
    def __init__(self, flt_model_path, fxp_model_path):
        self.flt_model = tf.keras.models.load_model(flt_model_path)
        self.fxp_model = tf.keras.models.load_model(fxp_model_path)
        assert len(self.flt_model.layers)==len(self.fxp_model.layers), "Number of layers should be equal!"

    def get_all_layers_out(self, model, input_tensor):
        input0 = model.layers[0].input
        outputs = [layer.output for layer in model.layers]
        intermediate_model = Model(inputs=input0, outputs=outputs)
        y = intermediate_model.predict(input_tensor)
        return y

    def get_layer_out(self, model, layer, input_tensor):
        input0 = model.layers[0].input
        intermediate_model = Model(inputs=input0, outputs=layer.output)
        y = intermediate_model.predict(input_tensor)
        return y

    def get_layer_in(self, model, layer, input_tensor):
        inLayers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inLayers, list): # make inLayers a list
            inLayers = [inLayers]
        if len(inLayers)==0:
            return input_tensor
        x = []
        for inLayer in inLayers:
            y = self.get_layer_out(model, inLayer, input_tensor)
            x.append(y)
        return x

    def single_layer_out(self, layer, input_tensor):
        intermediate_model = Model(inputs=layer.input, outputs=layer.output)
        y = intermediate_model.predict(input_tensor)
        return y

    def MSE(self, x1, x2, epsilon=1e-10):
        x1_flatten = np.ravel(x1)
        x2_flatten = np.ravel(x2)
        return 10*np.log10(np.mean(np.square(x2_flatten-x1_flatten)) + epsilon)

    def MaxErr(self, x1, x2, epsilon=1e-10):
        x1_flatten = np.ravel(x1)
        x2_flatten = np.ravel(x2)
        return 10*np.log10(np.max(np.square(x2_flatten-x1_flatten)) + epsilon)

    def WorstN(self, maxErrors, N=4):
        maxErrorsArray = np.array(maxErrors)
        indexs = np.argsort(-maxErrorsArray)[:N]
        return indexs

    def hist(self, x1, x2):
        x1_flatten = np.ravel(x1)
        x2_flatten = np.ravel(x2)
        hist1, bin_edges = np.histogram(x1_flatten, bins=100, density=True)
        hist2, _ = np.histogram(x2_flatten, bins=100, density=True)
        return [hist1, hist2, bin_edges[1:]]

    def non_injection(self, input_tensor):
        out1 = self.get_all_layers_out(self.flt_model, input_tensor)
        out2 = self.get_all_layers_out(self.fxp_model, input_tensor)
        return out1, out2

    def injection(self, input_tensor):
        out1 = self.get_all_layers_out(self.flt_model, input_tensor)
        layers1 = self.flt_model.layers
        layers2 = self.fxp_model.layers
        out2 = []
        for i, layer1 in tqdm(enumerate(layers1)):
            layer2 = layers2[i]
            x = self.get_layer_in(self.flt_model, layer1, input_tensor)
            y = self.single_layer_out(layer2, x)
            out2.append(y)
        return out1, out2

    def compare(self, input0):
        # Get activations after each layer
        out1, out2 = self.injection(input0)
        # -------------------------
        # Calculate mertrics
        # -------------------------
        # MSE, MaxError
        MSEs, MAXs = [], []
        for x1, x2 in zip(out1,out2):
            MSEs.append(self.MSE(x1, x2))
            MAXs.append(self.MaxErr(x1, x2))
        self.plot_errors(MSEs, MAXs)

        # Histogram
        layerIndexs = self.WorstN(MSEs)
        histograms =[]
        for layerIndex in layerIndexs:
            histogram = self.hist(out1[layerIndex], out2[layerIndex])
            histograms.append(histogram)
        self.plot_hist(histograms, layerIndexs)

        # Samples
        self.plot_samples(out1, out2, layerIndexs)


    def plot_errors(self, MSEs, MaxErrs):
        fig0, axes0 = plt.subplots(2, 2, constrained_layout=True)
        fig0.suptitle('MSE_MaxError')
        axes0[0,0].set_title("MSE_MaxError")
        axes0[0,0].set_xlabel('layer')
        axes0[0,0].set_ylabel('dB')
        axes0[0,0].grid()
        axes0[0,0].axis('auto')
        axes0[0,0].plot(MSEs, label="MSE")
        axes0[0,0].plot(MaxErrs, label="MaxError")
        axes0[0,0].legend(loc='lower right', prop={'size': 10})

    def plot_hist(self, histograms, layerIndexs):
        fig0, axes0 = plt.subplots(2, 2, constrained_layout=True)
        fig0.suptitle('Histogram')
        for i in range(2):
            for j in range(2):
                index = 2*i+j
                layerIndex = layerIndexs[index]
                hist1, hist2, bin_edges = histograms[index]
                axes0[i,j].set_title(self.fxp_model.layers[layerIndex].name+"[{}]".format(layerIndex))
                axes0[i,j].set_xlabel('value')
                axes0[i,j].set_ylabel('Probability')
                axes0[i,j].grid()
                axes0[i,j].axis('auto')
                axes0[i,j].plot(bin_edges, hist1, label="flp")
                axes0[i,j].plot(bin_edges, hist2, label="fxp")
                axes0[i,j].legend(loc='lower right', prop={'size': 10})

    def plot_samples(self, out1, out2, layerIndexs):
        fig0, axes0 = plt.subplots(2, 2, constrained_layout=True)
        fig0.suptitle('Samples')
        for i in range(2):
            for j in range(2):
                index = 2*i+j
                layerIndex = layerIndexs[index]
                axes0[i,j].set_title(self.fxp_model.layers[layerIndex].name+"[{}]".format(layerIndex))
                axes0[i,j].set_xlabel('sample')
                axes0[i,j].set_ylabel('value')
                axes0[i,j].grid()
                axes0[i,j].axis('auto')
                y1 = np.ravel(out1[layerIndex])[::1000]
                axes0[i,j].plot(y1, label="flp")
                y2 = np.ravel(out2[layerIndex])[::1000]
                axes0[i,j].plot(y2, label="fxp")
                axes0[i,j].legend(loc='lower right', prop={'size': 10})

if __name__ == "__main__":
    image_size = (640, 640, 3)
    flp_path = "/data/projects/swat/users/eladco/models/yolov5n_q16.h5"
    fxp_path = "/data/projects/swat/users/eladco/models/yolov5n_q.h5"

    #input0 = tf.random.uniform((1, *image_size), seed=2022)
    #input0 = tf.random.normal((1, *image_size), mean=0.5, stddev=1/12, seed=2022)
    #input0 = tf.ones((1, *image_size))

    #input0 = cv2.cvtColor(cv2.imread("../../data/images/bus.jpg"), cv2.COLOR_BGR2RGB)
    input0 = cv2.cvtColor(cv2.imread("../../data/images/zidane.jpg"), cv2.COLOR_BGR2RGB)
    input0 = cv2.resize(input0, (640,640))
    input0 = letterbox(input0, new_shape=(640,640)) / 255.0
    input0 = input0[np.newaxis, ...]

    modelFxpCompare = ModelFxpCompare(flp_path, fxp_path)
    modelFxpCompare.compare(input0)
    plt.show()

