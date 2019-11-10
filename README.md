# Image-captioning

## Overview
In this project, a CNN-LSTM encoder-decoder model was used to generate captions for images automatically. A complex deep learning model is used comprising of two components: a Convolutional Neural Network (CNN) that transforms an input image into a set of features - encoding the information in an image into a vector (known as embedding, which is a featurized representation of the image), and an RNN that turns those features into descriptive language.

<img src='images/intro.PNG' width=100% height=100%/>

## File description
```Notebook 0:``` Initializes the COCO API (the "pycocotools" library), used to access data from the MS COCO (Common Objects in Context) dataset. <br/>
```Notebook 1:``` uses the pycocotools, torchvision transforms, and NLTK to preprocess the images and the captions for network training. It also explores the EncoderCNN and DecoderRNN. <br/>
```model.py:``` Architecture and ```forward()``` functions for encoder and decoder. <br/>
```Notebook 2:``` Selection of hyperparameter values and training. The hyperparameter selections are explained. <br/>
```Notebook 3:``` Using the trained model to generate captions for images in the test dataset. <br/>
```data_loader.py:``` Custom data loader for PyTorch combining the dataset and the sampler. <br/>
```vocabulary.py:``` Vocabulary constructor built from the captions in the training dataset. <br/>

## Encoder-Decoder model
<img src='images/encoder-decoder.png' width=100% height=100%/><br/>
This is the complete architecture of the image captioning model. The CNN encoder basically finds patterns in images and encodes it into a vector that is passed to the LSTM decoder that outputs a word at each time step to best describe the image. Upon reaching the ```<end>``` token, the entire caption is generated and that is our output for that particular image.

### Encoder:
<img src='images/encoder.png' width=70% height=70%/> <br/>
The encoder is based on a Convolutional Neural Network that encodes an image into a featurized compact representation (in the form of an embedding).
The CNN-Encoder is a ResNet (Residual Network). These kind of networks help in diminishing the vanishing and exploding gradient problems. I used the ResNet-152 pre-trained model.

### Decoder: 
<img src='images/decoder.png' width=70% height=70%/> <br/>
The CNN encoder is followed by a recurrent neural network (LSTM) that generates a corresponding sentence. <br/>
The feature vector is fed into the "DecoderRNN"  (which is "unfolded" in time). Each word appearing as output at the top is fed back to the network as input (at the bottom) in a subsequent time step, until the entire caption is generated. The arrow pointing to the right that connects the LSTM boxes together represents hidden state information, which represents the network's "memory", also fed back to the LSTM at each time step.

## Results:
<img src='images/train.PNG' width=50% height=50%/> <br/>
<img src='images/street.PNG' width=50% height=50%/> <br/>
<img src='images/ski.PNG' width=50% height=50%/> <br/>

### Funny fails:
<img src='images/darth.PNG' width=50% height=50%/> <br/>
<img src='images/cake.PNG' width=50% height=50%/> <br/>

