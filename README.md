# Art-Style-Transfer-using-TensorFlow
Neural Art Style Transfer based on [Gatys et al.](https://arxiv.org/abs/1508.06576)

**Contents**
- About
- Classes
- Scripts
- Executables
- Outputs
  1. Columbia University Low Library - Van Gogh Starry Night
  2. Mozart - Van Gogh - Portrait
  3. Horse - Kandinsky

**About**</br>
This is a fun project that I have undertaken because of my fascination in style transfer techniques. The project is based on the original [Gatys et al.](https://arxiv.org/abs/1508.06576) paper.
Some of the code for the project is inspired from the TensorFlow tutorials. My inspiration for this project is from the "Convolutional Neural Networks" course offered by deeplearning.ai on Coursera taught by Dr. Andrew NG.
Although more sophisticated techniques and algorithms for neural style transfer are available today, I wanted to try these techniques for myself and visualize the resulting stylized outputs.

**Classes**
- BuildModel : Class to build the VGG model and extract layers and outputs of content and style images processed from the VGG model and Gram matrix
- DefineRepresentations : Class to initialize the style and content layers from the VGG19 model

**Scripts**
- ImagePreprocessing : Several utility and helper functions for preprocessing, scaling, loading, and display of input and output images

**Executables**
- TrainModel : Regularizing input images, defining loss function, defining training step and hyperparametrs, and training to get stylized output image

**Outputs**
- **Columbia University Low Library - Van Gogh Starry Night**
  - Parameters :
    1. Total variation weight = 30
    2. Style weight = 5e-2
    3. Content weight = 1e4
    4. Epochs = 25
    5. Steps per epoch = 100
    6. optimizer = Adam(learning rate = 0.005, beta_1 = 0.99, epsilon = 1e-1)
  - Input Images :
    - Content Image :
      <img src="Images/source/content.jpg" width=1000>
    - Style Image :
      <img src="Images/source/style.jpg" width=1000>
  - Rescaled Images :
    <img src="Images/outputs/resaled_content_and_style_images.png" width=1000>
  - Output Images :<br />
    Epoch 1 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; &nbsp; &nbsp; Epoch 5 <br />
    <img src="Images/outputs/epoch_1.png" width=350> &nbsp; &nbsp; &nbsp; &nbsp; <img src="Images/outputs/epoch_5.png" width=350> <br /><br />
    Epoch 10 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; &nbsp; Epoch 15 <br />
    <img src="Images/outputs/epoch_10.png" width=350> &nbsp; &nbsp; &nbsp; &nbsp; <img src="Images/outputs/epoch_15.png" width=350> <br /><br />
    Epoch 20 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; &nbsp; Epoch 25 <br />
    <img src="Images/outputs/epoch_20.png" width=350> &nbsp; &nbsp; &nbsp; &nbsp; <img src="Images/outputs/epoch_25.png" width=350> <br /><br />
  - Stylized Image : <br />
    <img src="Images/outputs/stylized-image.png" width=1000>
- **Mozart - Van Gogh - Portrait**
  - Parameters :
    1. Total variation weight = 30
    2. Style weight = 5e-2
    3. Content weight = 1e4
    4. Epochs = 25
    5. Steps per epoch = 100
    6. optimizer = Adam(learning rate = 0.005, beta_1 = 0.99, epsilon = 1e-1)
  - Input Images :
    - Content Image : <br />
      <img src="Images/other_examples/1/content.jpg" Height=500>
    - Style Image : <br />
      <img src="Images/other_examples/1/style.jpg" Height=500>
  - Rescaled Images : <br />
    <img src="Images/other_examples/1/resaled_content_and_style_images.png" width=1000>
  - Stylized Image : <br />
    <img src="Images/other_examples/1/stylized-image.png" height=1000>
- **Horse - Kandinsky**
  - Parameters :
    1. Total variation weight = 30
    2. Style weight = 5e-2
    3. Content weight = 1e4
    4. Epochs = 25
    5. Steps per epoch = 100
    6. optimizer = Adam(learning rate = 0.005, beta_1 = 0.99, epsilon = 1e-1)
  - Input Images :
    - Content Image : <br />
      <img src="Images/other_examples/2/content.jpg" width=1000>
    - Style Image : <br />
      <img src="Images/other_examples/2/style.jpg" width=1000>
  - Rescaled Images : <br />
    <img src="Images/other_examples/2/resaled_content_and_style_images.png" width=1000>
  - Stylized Image : <br />
    <img src="Images/other_examples/2/stylized-image.png" width=1000>
