"""
BuildModel:
    1) Initializes Class StyleContentExtraction
"""

import numpy as np
import tensorflow as tf
import ImagePreprocessing as IP
import DefineRepresentations as DR
import logging
from ImagePreprocessing import VggModelLayers, LayerStats

CONTENT_IMAGE = IP.content_img
STYLE_IMAGE = IP.style_img
LAYER_REP = DR.RepresentationLayers()
CONTENT_LAYERS, STYLE_LAYERS = LAYER_REP()

class StyleContentExtraction(tf.keras.models.Model):
    
    """
    Class to extract layers and outputs of content and style images processed from the VGG model and Gram matrix
    """

    def __init__(self, style_layers, content_layers, style_image=None, content_image=None, 
                    style_stats=False, content_stats=False):
        """
        Inputs:
            style_layers: chosen style layers from the VGG model
            content_layers: chosen content layers from the VGG model
            style_image: loaded style image defaults to None
            content_image: loaded content image defaults to None
            style_stats: boolean to log style layers stats
            content_stats: boolean to stats content layer stats

        Actions:
            1) Initializes the VGG model from the style and content layers using
               the imported helper function - VggModelLayers
            2) Initializes the style and content layers from the inputs
            3) Initializes the number of style layers from style layers
            4) Freezes the VGG layers training
            5) Log style and content layer stats according to the input stat booleans
        """

        super(StyleContentExtraction, self).__init__()
        logging.basicConfig(format="%(message)s", level=logging.INFO)

        self.vgg = VggModelLayers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

        # extracting style layers
        if style_stats:
            style_extractor = VggModelLayers(style_layers)
            if style_image != None:
                style_outputs = style_extractor(style_image * 255)
                # style layers stats
                LayerStats(style_layers, style_outputs)

        if content_stats:
            content_extractor = VggModelLayers(content_layers)
            if content_image != None:
                content_outputs = content_extractor(content_image * 255)
                # content layer stats
                LayerStats(content_layers, content_outputs)

    # calculating style using gram matrix
    def GramMatrix(self, tensor):
        """
        Input:
            tensor: input tensor (of style image outputs from VGG model in this case)
        
        Outputs:
            Gram matrix of the input tensor
        """
        output = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
        input_shape = tf.shape(tensor)
        locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return output/(locations)

    def call(self, inputs):
        """
        Inputs:
            inputs: normalized image tensor containing values of float in [0, 1]
        
        Outputs:
            A dictionary mapping 'content' and 'style' to respective  zipped (layers and outputs) dictionaries
        """
        inputs = inputs * 255.0
        preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [self.GramMatrix(style_output) for style_output in style_outputs]
        content_dict = {name:value for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name:value for name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style':style_dict}
