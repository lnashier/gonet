## Train Network to learn a XOR function

```shell
go run . build xor
```

## Train Network to learn an OR function

```shell
go run . build or
```

## Train Network to learn image classes

Before we do

### Would Feedforward Neural network work for image classification?

A basic feedforward neural network, which can be used for simple tasks such as the XOR function or other basic
classification tasks is not suitable for tasks for predicting image classes directly.

In the tasks like image classification, we would typically use convolutional neural networks (CNNs). CNNs are
specifically designed to handle image data efficiently by exploiting the spatial structure of the images. They consist
of convolutional layers followed by pooling layers, and they are widely used in image-related tasks.

While we could technically use a feedforward neural network for image prediction by flattening the images into
one-dimensional arrays and feeding them into the network, the performance would likely be poor compared to CNNs,
especially in image classification where spatial relationships are important.

If we want to work with image data, I recommend looking into CNN architectures and libraries/frameworks that support
them. These frameworks provide high-level APIs for building and training CNNs and have pre-built models that we can use
or adapt for specific tasks.

## Image classification with Feedforward network, how would we do that?

Using the simple feedforward neural network for image classification would involve some preprocessing steps to flatten
the images and prepare them as input for the network. However, it's important to note that this approach may not yield
satisfactory results compared to using convolutional neural networks (CNNs), which are better suited for image
classification tasks due to their ability to capture spatial relationships in images.

Nevertheless, we want to proceed with using the simple feedforward neural network for image classification as an
exercise, here's a general outline of how we would approach it:

1. **Data Preprocessing**: Flatten the images and prepare them as input vectors for the neural network. Depending on the
   size of the images and the input size of the neural network, we may need to resize or down sample the images.

2. **Training**: Train the neural network using the flattened image data and their corresponding labels. This involves
   passing the flattened image vectors through the network, computing the loss, and updating the network parameters
   using backpropagation.

3. **Evaluation**: Evaluate the performance of the trained neural network on a separate test dataset. This involves
   passing the test images through the network, computing predictions, and comparing them to the ground truth labels to
   measure accuracy or other evaluation metrics.

### Train the network to learn MNIST

[MNIST Wiki](https://en.wikipedia.org/wiki/MNIST_database)

[Download](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

In this example, we're using MNIST dataset. The input images are 28x28 grayscale images. We flatten each image into a 1D
array of length 784 (28x28), which serves as the input to the neural network. The output size is set to 10, corresponding
to the 10 classes of digits (0-9) in the MNIST dataset.

```shell
go run . build mnist bin/data/mnist/train-images-idx3-ubyte.gz bin/data/mnist/train-labels-idx1-ubyte.gz bin/data/mnist/t10k-images-idx3-ubyte.gz bin/data/mnist/t10k-labels-idx1-ubyte.gz
```
