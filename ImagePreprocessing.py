"""
ImagePreprocessing:
    1) Defines several helper functions to be used globally
    2) Initializes the content and style images
    3) Saves the images to the output directory
"""

import PIL.Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['axes.grid'] = False
import logging

# logging configuration
logging.basicConfig(format="%(message)s", level=logging.INFO)

# image directory paths
content_image_path = "Images/source/content.jpg"
style_image_path = "Images/source/style.jpg"

""" helper functions """
def tensorToImage(tensor):
    """
    Input:
        tensor: Input tensor
    
    Output:
        Returns image from the input tensor
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# function to load image with set max dimention
def loadImage(path, maxdim=600):
    """
    Input:
        path: directory path of image to load

    Outputs:
        Loads and returns the image from path
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    longdim = max(shape)
    scale = maxdim / longdim

    newShape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, newShape)
    image = image[tf.newaxis, :]
    return image

# function to display images
def imshow(image, title=None):
    """
    Input:
        image: image to show
        title: title for image
    
    Output:
        displays the input image
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

def VggModelLayers(layer_names):
    """
    Input:
        layer_names: layers to be included in the model
    
    Output:
        Creates and returns a vgg model that returns a list of intermediate output values
    """
    # loading a pre-trained VGG model trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # freezing vgg layers
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# function to represent staticstics for each layer's output
def LayerStats(layers, outputs, verbose=False):
    """
    Input:
        layers: list of layers
        outputs: outputs of the layers
        verbose: bollean to turn on logging info
    
    Output:
        Logs the stats of layers and outputs from the inputs
    """
    for name, output in zip(layers, outputs):
        if verbose:
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


""" displaying images"""
content_img = loadImage(content_image_path)
style_img = loadImage(style_image_path)

plt.subplot(1, 2, 1)
imshow(content_img, "Content Image")

plt.subplot(1, 2, 2)
imshow(style_img, "Style Image")

plt.savefig("Images/outputs/resaled_content_and_style_images.png")
#plt.show()
