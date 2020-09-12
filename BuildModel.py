import numpy as np
import tensorflow as tf
import ImagePreprocessing as IP
import DefineRepresentations as DR
import logging

CONTENT_IMAGE = IP.content_img
STYLE_IMAGE = IP.style_img
CONTENT_LAYERS = DR.content_layers
STYLE_LAYERS = DR.style_layers

'''
class StyleContentExtraction(tf.keras.models.Model):
    
    def __init__(self, style_layers, content_layers, style_image=None, content_image=None, 
                    style_stats=False, content_stats=False):

        super(StyleContentExtraction, self).__init__()
        logging.basicConfig(format="%(message)s", level=logging.INFO)

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
                log_name = name
                shape_log = "    shape: " + str(output.numpy().shape)
                min_log = "    min: " + str(output.numpy().min())
                max_log = "    max: " + str(output.numpy().max())
                mean_log = "    mean: " + str(output.numpy().mean())
                nextline = "\n"
                logging.info(log_name)
                logging.info(shape_log)
                logging.info(min_log)
                logging.info(max_log)
                logging.info(mean_log)
                logging.info(nextline)
                """
                print(name)
                print("     shape: ", output.numpy().shape)
                print("     min: ", output.numpy().min())
                print("     max: ", output.numpy().max())
                print("     mean: ", output.numpy().mean())
                print()
                """
        
        self.vgg = VggModelLayers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

        # extracting style layers
        if style_stats:
            style_extractor = VggModelLayers(style_layers)
            assert(style_image)
            style_outputs = style_extractor(style_image * 255)
            # style layers stats
            LayerStats(style_layers, style_outputs)

        if content_stats:
            content_extractor = VggModelLayers(content_layers)
            assert(content_image)
            content_outputs = content_extractor(content_image * 255)
            # content layer stats
            LayerStats(content_layers, content_outputs)

    # calculating style using gram matrix
    def GramMatrix(tensor):
        output = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
        input_shape = tf.shape(tensor)
        locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return output/(locations)

    def call(self, inputs):
        """Input: float in [0,1]"""
        inputs = inputs * 255.0
        preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [GramMatrix(style_output) for style_output in style_outputs]
        content_dict = {name:value for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name:value for name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style':style_dict}
'''


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
#style_extractor = VggModelLayers(STYLE_LAYERS)
#style_outputs = style_extractor(STYLE_IMAGE * 255)

# style layers stats
#LayerStats(STYLE_LAYERS, style_outputs)

# calculating style using gram matrix
def GramMatrix(tensor):
    output = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return output/(locations)

# extracting style and content
class StyleContentExtractionModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentExtractionModel, self).__init__()
        self.vgg = VggModelLayers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Input: float in [0,1]"""
        inputs = inputs * 255.0
        preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [GramMatrix(style_output) for style_output in style_outputs]
        content_dict = {name:value for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name:value for name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style':style_dict}


# setting feature extractor
extractor = StyleContentExtractionModel(STYLE_LAYERS, CONTENT_LAYERS)

