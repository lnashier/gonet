# Feedforward Examples

You may want to experiment with hyperparameters like the structure of hidden layers, epochs, and learning rate.

## Train Network to learn a XOR function

```shell
go run . build xor
```

```text
%go run . build xor
Shapes: [2 4 1]
Hidden Layers: 1

Epoch 0000, Loss: 0.249986
Epoch:(0) Inputs:(4) Duration:(30.625µs)
Epoch 10000, Loss: 0.000667
Epoch:(10000) Inputs:(4) Duration:(1.75µs)
Epoch 20000, Loss: 0.000303
Epoch:(20000) Inputs:(4) Duration:(5.5µs)
Epoch 30000, Loss: 0.000196
Epoch:(30000) Inputs:(4) Duration:(1.041µs)
Epoch 40000, Loss: 0.000145
Epoch:(40000) Inputs:(4) Duration:(1.083µs)
Epoch 50000, Loss: 0.000115
Epoch:(50000) Inputs:(4) Duration:(1.125µs)
Epoch 60000, Loss: 0.000095
Epoch:(60000) Inputs:(4) Duration:(958ns)
Epoch 70000, Loss: 0.000081
Epoch:(70000) Inputs:(4) Duration:(1µs)
Epoch 80000, Loss: 0.000071
Epoch:(80000) Inputs:(4) Duration:(1.125µs)
Epoch 90000, Loss: 0.000063
Epoch:(90000) Inputs:(4) Duration:(958ns)
TrainingDuration 157.754042ms
[0 0] -> [0.008840288940971571]
[0 1] -> [0.9928844923165359]
[1 0] -> [0.9922301318809412]
[1 1] -> [0.00597337887472345]
```
## Train Network to learn an OR function

```shell
go run . build or
```

```text
%go run . build or
Shapes: [2 4 1]
Hidden Layers: 1

Epoch 0000, Loss: 0.247073
Epoch:(0) Inputs:(4) Duration:(12µs)
Epoch 100000, Loss: 0.001009
Epoch:(100000) Inputs:(4) Duration:(1.25µs)
Epoch 200000, Loss: 0.000425
Epoch:(200000) Inputs:(4) Duration:(1.25µs)
Epoch 300000, Loss: 0.000265
Epoch:(300000) Inputs:(4) Duration:(1.084µs)
Epoch 400000, Loss: 0.000192
Epoch:(400000) Inputs:(4) Duration:(1.208µs)
Epoch 500000, Loss: 0.000150
Epoch:(500000) Inputs:(4) Duration:(1.084µs)
Epoch 600000, Loss: 0.000123
Epoch:(600000) Inputs:(4) Duration:(1.084µs)
Epoch 700000, Loss: 0.000104
Epoch:(700000) Inputs:(4) Duration:(1.208µs)
Epoch 800000, Loss: 0.000090
Epoch:(800000) Inputs:(4) Duration:(1.625µs)
Epoch 900000, Loss: 0.000079
Epoch:(900000) Inputs:(4) Duration:(1.542µs)
TrainingDuration 1.3648915s
[0 0] -> [0.01301274172801166]
[0 1] -> [0.9925989889062432]
[1 0] -> [0.9926364971902891]
[1 1] -> [0.9976853130924934]
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

```text
% go run . build mnist bin/data/mnist/train-images-idx3-ubyte.gz bin/data/mnist/train-labels-idx1-ubyte.gz bin/data/mnist/t10k-images-idx3-ubyte.gz bin/data/mnist/t10k-labels-idx1-ubyte.gz
Shapes: [784 128 128 10]
Hidden Layers: 2

Epoch:(0) Inputs:(20129) Duration:(5.000996208s)
Epoch:(0) Inputs:(40106) Duration:(10.000954542s)
Epoch 0000, Loss: 0.108233
Epoch:(0) Inputs:(60000) Duration:(14.995144s)
Epoch:(1) Inputs:(4031) Duration:(1.012468708s)
Epoch:(1) Inputs:(24013) Duration:(6.012457583s)
Epoch:(1) Inputs:(43934) Duration:(11.0124425s)
Epoch 0001, Loss: 0.073246
Epoch:(1) Inputs:(60000) Duration:(15.008577291s)
Epoch:(2) Inputs:(8163) Duration:(2.024356208s)
Epoch:(2) Inputs:(28190) Duration:(7.024828625s)
Epoch:(2) Inputs:(48099) Duration:(12.024849416s)
Epoch 0002, Loss: 0.059170
Epoch:(2) Inputs:(60000) Duration:(15.011319791s)
Epoch:(3) Inputs:(12313) Duration:(3.081071708s)
Epoch:(3) Inputs:(32258) Duration:(8.081528208s)
Epoch:(3) Inputs:(52250) Duration:(13.081527167s)
Epoch 0003, Loss: 0.050646
Epoch:(3) Inputs:(60000) Duration:(15.021553375s)
Epoch:(4) Inputs:(16245) Duration:(4.054449958s)
Epoch:(4) Inputs:(36199) Duration:(9.053939542s)
Epoch:(4) Inputs:(56267) Duration:(14.054472167s)
Epoch 0004, Loss: 0.048464
Epoch:(4) Inputs:(60000) Duration:(14.986163833s)
Epoch:(5) Inputs:(452) Duration:(113.939208ms)
Epoch:(5) Inputs:(20391) Duration:(5.114458208s)
Epoch:(5) Inputs:(40379) Duration:(10.114481958s)
Epoch 0005, Loss: 0.052027
Epoch:(5) Inputs:(60000) Duration:(15.00363925s)
Epoch:(6) Inputs:(4639) Duration:(1.148865s)
Epoch:(6) Inputs:(24796) Duration:(6.148887209s)
Epoch:(6) Inputs:(44928) Duration:(11.148865292s)
Epoch 0006, Loss: 0.043205
Epoch:(6) Inputs:(60000) Duration:(14.896234792s)
Epoch:(7) Inputs:(9326) Duration:(2.311359334s)
Epoch:(7) Inputs:(29482) Duration:(7.311384625s)
Epoch:(7) Inputs:(49487) Duration:(12.311361s)
Epoch 0007, Loss: 0.049056
Epoch:(7) Inputs:(60000) Duration:(14.918633667s)
Epoch:(8) Inputs:(13392) Duration:(3.365757s)
Epoch:(8) Inputs:(33531) Duration:(8.365755125s)
Epoch:(8) Inputs:(53614) Duration:(13.36575325s)
Epoch 0008, Loss: 0.047194
Epoch:(8) Inputs:(60000) Duration:(14.958357833s)
Epoch:(9) Inputs:(17621) Duration:(4.464648334s)
Epoch:(9) Inputs:(37194) Duration:(9.46462125s)
Epoch:(9) Inputs:(57328) Duration:(14.464644s)
Epoch 0009, Loss: 0.040516
Epoch:(9) Inputs:(60000) Duration:(15.129230125s)
TrainingDuration 3m59.579139625s
Total Predictions: 10000, Correct Predictions: 9676, Accuracy: 96.76%
```


```text
% go run . build mnist bin/data/mnist/train-images-idx3-ubyte.gz bin/data/mnist/train-labels-idx1-ubyte.gz bin/data/mnist/t10k-images-idx3-ubyte.gz bin/data/mnist/t10k-labels-idx1-ubyte.gz
Shapes: [784 128 10]
Hidden Layers: 1

Epoch:(0) Inputs:(25859) Duration:(5.000984334s)
Epoch:(0) Inputs:(51701) Duration:(10.001020875s)
Epoch 0000, Loss: 0.112682
Epoch:(0) Inputs:(60000) Duration:(11.606173334s)
Epoch:(1) Inputs:(3804) Duration:(734.183208ms)
Epoch:(1) Inputs:(29710) Duration:(5.734159833s)
Epoch:(1) Inputs:(55621) Duration:(10.734186125s)
Epoch 0001, Loss: 0.078596
Epoch:(1) Inputs:(60000) Duration:(11.579851167s)
Epoch:(2) Inputs:(7692) Duration:(1.4944955s)
Epoch:(2) Inputs:(33380) Duration:(6.494490333s)
Epoch:(2) Inputs:(59184) Duration:(11.494519375s)
Epoch 0002, Loss: 0.063419
Epoch:(2) Inputs:(60000) Duration:(11.654099791s)
Epoch:(3) Inputs:(11322) Duration:(2.183396875s)
Epoch:(3) Inputs:(37227) Duration:(7.183401875s)
Epoch 0003, Loss: 0.053483
Epoch:(3) Inputs:(60000) Duration:(11.578242s)
Epoch:(4) Inputs:(15273) Duration:(2.948743375s)
Epoch:(4) Inputs:(41146) Duration:(7.948716167s)
Epoch 0004, Loss: 0.046826
Epoch:(4) Inputs:(60000) Duration:(11.620672584s)
Epoch:(5) Inputs:(18728) Duration:(3.666232291s)
Epoch:(5) Inputs:(44184) Duration:(8.666230291s)
Epoch 0005, Loss: 0.041967
Epoch:(5) Inputs:(60000) Duration:(11.716188125s)
Epoch:(6) Inputs:(22169) Duration:(4.299435083s)
Epoch:(6) Inputs:(47965) Duration:(9.299437708s)
Epoch 0006, Loss: 0.038013
Epoch:(6) Inputs:(60000) Duration:(11.643503583s)
Epoch:(7) Inputs:(25898) Duration:(4.998766792s)
Epoch:(7) Inputs:(51664) Duration:(9.998798208s)
Epoch 0007, Loss: 0.034757
Epoch:(7) Inputs:(60000) Duration:(11.608739208s)
Epoch:(8) Inputs:(3832) Duration:(739.74675ms)
Epoch:(8) Inputs:(29726) Duration:(5.739744333s)
Epoch:(8) Inputs:(55600) Duration:(10.739747042s)
Epoch 0008, Loss: 0.032182
Epoch:(8) Inputs:(60000) Duration:(11.588483083s)
Epoch:(9) Inputs:(7619) Duration:(1.475462125s)
Epoch:(9) Inputs:(33531) Duration:(6.475455542s)
Epoch:(9) Inputs:(59436) Duration:(11.475477958s)
Epoch 0009, Loss: 0.030089
Epoch:(9) Inputs:(60000) Duration:(11.584600083s)
TrainingDuration 3m12.771996916s
Total Predictions: 10000, Correct Predictions: 9719, Accuracy: 97.19%
```