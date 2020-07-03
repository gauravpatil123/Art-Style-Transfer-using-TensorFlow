import numpy as np
import tensorflow as tf
import ImagePreprocessing as IP
import DefineRepresentations as DR

CONTENT_IMAGE = IP.content_img
STYLE_IMAGE = IP.style_img
CONTENT_LAYERS = DR.content_layers
STYLE_LAYERS = DR.style_layers

def VggModelLayers(layer_names):
    """creates and returns a vgg model 
    that returns a list of intermediate outut values"""
    # loading a pre-trained VGG model trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # freezing vgg layers
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# function to represent staticstics for each layer's output
def LayerStats(layers, outputs):
    for name, output in zip(layers, outputs):
        print(name)
        print("     shape: ", output.numpy().shape)
        print("     min: ", output.numpy().min())
        print("     max: ", output.numpy().max())
        print("     mean: ", output.numpy().mean())
        print()

# extracting style layers
style_extractor = VggModelLayers(STYLE_LAYERS)
style_outputs = style_extractor(STYLE_IMAGE * 255)

# style layers stats
#LayerStats(STYLE_LAYERS, style_outputs)

# calculating style using gram matrix
def GramMatrix(tensor):
    output = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return output/(locations)

