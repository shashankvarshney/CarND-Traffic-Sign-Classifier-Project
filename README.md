# Project: German Traffic Sign Classifier

# Table of Contents
- [Project Description](#project-description)
- [Environment and Programming language](#environment-and-programming-language)
- [Data Description](#data-description)
- [Data Reading and Exploration](#data-reading-and-exploration)
- [Design and Test a Model Architecture](#design-and-test-a-model-architecture)
  * [Preprocess the Data](#preprocess-the-data)
  * [Saving the preprocessed data](#saving-the-preprocessed-data)
  * [Shuffling the data](#shuffling-the-data)
- [Model Architecture](#model-architecture)
- [Train, Validate and Test the model](#train--validate-and-test-the-model)
- [Test the Model on new images](#test-the-model-on-new-images)
- [Further Improvements](#further-improvements)

## Project Description
In this project, we will use Convolution Neural Network(ConvNet) to classify the [German Traffic Signs]((http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Dataset has already been downloaded and saved in [data directory](../data/).

## Environment and Programming language
* Python 3.5.2 has been used.
* Miniconda framework has been used which can be downloaded from this [link](https://repo.continuum.io/miniconda/).
* Once installed, open conda prompt.
* Create virtual environment by using `conda env create -f environment.yml`. [**environment.yml**](./environment.yml) has been included in the repository. If GPU is available then use [**environment-gpu.yml**](./environment-gpu.yml) and use `conda env create -f environment-gpu.yml`
* Jupyter notebook has been used for interactively analyzing and exploring the data.
* Python code and calculation given in the [Notebook file](./CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb) file.
* Tensorflow framework is used for training the Convolutional Neural Network (CNN)

## Data Description
Data has already been downloaded and divided into 3 parts and has been saved into "data" directory:
* Training Data (train.p)
* Validation Data (valid.p)
* Testing Data (test.p)

Model will be fit on Training Data, validated on the Validation Data and will be finally tested on Testing Data and accuracy on testing data will be final accuracy.

## Data Reading and Exploration
Data was saved by using `pickle` library so I have used `pickle.load()` function to read the data.

All 3 data sets (training, validation and testing) were loaded and saved in their respective X and y values.

We have following observations from the basic analysis of data sets:
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

I plotted number of samples in each class for all 3 data sets. Following are the bar plots of the same:

![bar_plot_training.jpg](./CarND-Traffic-Sign-Classifier-Project/bar_plot_training.jpg)
![bar_plot_validation.jpg](./CarND-Traffic-Sign-Classifier-Project/bar_plot_validation.jpg)
![bar_plot_test.jpg](./CarND-Traffic-Sign-Classifier-Project/bar_plot_test.jpg)

Following are the observations from all 3 bar plots:
* Distribution of data among various classes are almost similar in Train, Validation and testing data set.
* Class distribution is very unbalanced like for some classes we have 10 times more data as compared to some other classes.
* Classes which have less number of training samples can introduce some errors.
* Data augmentation is required to increase number of data points per class.

Finally one image from each class has been plotted in the ipython file. We can clearly see that some of the images are very dark and sign is not visible at all.

## Design and Test a Model Architecture
There are various aspects to consider when thinking about this problem:
- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

### Preprocess the Data
As the first step, its good that data should be normalized. Normalized data is helpful in faster convergence of the Gradient Descent.

Each image is normalized using min-max scaling. I used formula `(value - min)/(max - min)` and values of pixels in the image is changed between 0 to 1. Function `preprocess(image_data, image_label)` has the formula of min-max scaling.

All three datasets were preprocessed by using min-max scaler.

### Saving the preprocessed data
Preprocessed data can be pickeled and saved to local disk and preprocessed data can be loaded. So this can work as first checkpoint of the model.

Functions `save_data(filename, dictionary)` and `load_data(filename)` are used to save and load the preprocessed data sets. While saving the data, we used a dictionary for each data set so that data and labels can be saved together.

### Shuffling the data
Further training data should be shuffled so that we can use stochastic gradient descent (SGD) with batched data.

## Model Architecture
I used LeNet 5 model for training. Following is the architecture of the LeNet which has been used. I also used dropout layers also because model was overfitting the data as I was getting around 99.9% accuracy on Training set but only around 90% on Validation set.

Following is the architecture of the LeNet network:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5x3x6   | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Dropout	      	    | With keep probability as 0.7 during training 	|
| Convolution 5x5x6x16	| 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				    |
| Dropout	      	    | With keep probability as 0.7 during training 	|
| Fully connected		| Flatten layer with 400 input and 120 output   |
| RELU					|												|
| Fully connected		| Flatten layer with 120 input and 84 output    |
| RELU					|												|
| Fully connected		| Flatten layer with 84 input and 43 output     |
| Softmax				|         									    |

Following are the detailed building blocks of the above mentioned network:
1. Following is the first layer of the model.
```
conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
conv1_b = tf.Variable(tf.zeros(6))
conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
conv1 = tf.nn.dropout(conv1, keep_prob)
```
  * First Weights are initialized and a filter of shape (5, 5, 3, 6) is used where each element corresponds to (batch, height, width, channels).
  * Biases are initialized to all 0 values.
  * `tf.nn.conv2d()` is used for the convolution layer with strides [1, 1, 1, 1] corresponding to (batch, height, width, channels) with *valid padding*.
  * `tf.nn.relu()` is used for the *relu* activation.
  * Further `tf.nn.max_pool()` is used for max pooling which reduces the dimensions of the data.
  * Finally `tf.nn.dropout()` is used to dropout with value of set values of keep_prob.
2. Then second convolution layer is implemented in similar way.
```
conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
conv2_b = tf.Variable(tf.zeros(16))
conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
conv2 = tf.nn.dropout(conv2, keep_prob)
```
3. Then third convolution layer is implemented
```
conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
conv2_b = tf.Variable(tf.zeros(16))
conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
conv2 = tf.nn.dropout(conv2, keep_prob)
```
**Each convolution reduces the height and width of the image but increases the depth.**
4. The data is flatten to perform the normal neural network and fully connected layer.
```
fc0 = flatten(conv2)
fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
fc1_b = tf.Variable(tf.zeros(120))
fc1   = tf.matmul(fc0, fc1_W) + fc1_b
fc1    = tf.nn.relu(fc1)
```
5. Then second fully connected layer is implemented.
```
fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
fc2_b  = tf.Variable(tf.zeros(84))
fc2    = tf.matmul(fc1, fc2_W) + fc2_b
fc2    = tf.nn.relu(fc2)
```
6. Finally Output layer is implemented which returns logits on which softmax can be implemented.
```
fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
fc3_b  = tf.Variable(tf.zeros(43))
logits = tf.matmul(fc2, fc3_W) + fc3_b
```

**All of the above mentioned steps are implemented within a function `LeNet(x)` which takes data as input which is the input layer of the model.**

## Train, Validate and Test the model
A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

To train we need following parameters:
* Cost function which needs to be minimized. `cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)` gives that cost function.
* Loss operation to minimize the cost function. `loss_operation = tf.reduce_mean(cross_entropy)` gives the loss operation.
* Optimizer to perform the optimization. We can use normal gradient descent or Adam optimizer. Adam optimizer is used which in turn uses momentum and running average and take care of lot of hyperparameters automatically and perform optimization faster. `optimizer = tf.train.AdamOptimizer(learning_rate = rate)` performs the optimization.
* Finally a training operation which takes the optimizer and perform training with loss operation. `training_operation = optimizer.minimize(loss_operation)` performs the training operation.
* We also need to have accuracy operation to calculate the accuracy after each epoch. We use following code for the same.
```
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
* We also need a function to evaluate the accuracy. While calculating the accuracy we need to keep all the data so we need to use keep_prob = 1. Following function is used for the same:
```
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```
* Finally we train the model and evaluate it on training and validation set to check if any overfitting or underfitting case. Following code is used for this which ultimately uses `training_operation` and function `evaluate()` to train the model on training the data and also saves the model so that same model can be implemented on test set and unseen data later on without re-training the model which is a time consuming process.
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train, random_state = 1)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:0.7})

        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid_pre, y_valid_pre)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")
```
* I ran the model till 30 epochs and following is the outcome of the model training and evaluation:
```
Training...
EPOCH 1 ...
Training Accuracy = 0.807
Validation Accuracy = 0.742
EPOCH 2 ...
Training Accuracy = 0.915
Validation Accuracy = 0.851
EPOCH 3 ...
Training Accuracy = 0.948
Validation Accuracy = 0.885
EPOCH 4 ...
Training Accuracy = 0.959
Validation Accuracy = 0.892
EPOCH 5 ...
Training Accuracy = 0.969
Validation Accuracy = 0.908
EPOCH 6 ...
Training Accuracy = 0.975
Validation Accuracy = 0.915
EPOCH 7 ...
Training Accuracy = 0.979
Validation Accuracy = 0.918
EPOCH 8 ...
Training Accuracy = 0.981
Validation Accuracy = 0.927
EPOCH 9 ...
Training Accuracy = 0.986
Validation Accuracy = 0.927
EPOCH 10 ...
Training Accuracy = 0.986
Validation Accuracy = 0.930
EPOCH 11 ...
Training Accuracy = 0.988
Validation Accuracy = 0.932
EPOCH 12 ...
Training Accuracy = 0.987
Validation Accuracy = 0.913
EPOCH 13 ...
Training Accuracy = 0.991
Validation Accuracy = 0.942
EPOCH 14 ...
Training Accuracy = 0.990
Validation Accuracy = 0.935
EPOCH 15 ...
Training Accuracy = 0.992
Validation Accuracy = 0.937
EPOCH 16 ...
Training Accuracy = 0.993
Validation Accuracy = 0.938
EPOCH 17 ...
Training Accuracy = 0.993
Validation Accuracy = 0.940
EPOCH 18 ...
Training Accuracy = 0.994
Validation Accuracy = 0.940
EPOCH 19 ...
Training Accuracy = 0.995
Validation Accuracy = 0.939
EPOCH 20 ...
Training Accuracy = 0.995
Validation Accuracy = 0.939
EPOCH 21 ...
Training Accuracy = 0.995
Validation Accuracy = 0.949
EPOCH 22 ...
Training Accuracy = 0.995
Validation Accuracy = 0.939
EPOCH 23 ...
Training Accuracy = 0.996
Validation Accuracy = 0.943
EPOCH 24 ...
Training Accuracy = 0.997
Validation Accuracy = 0.939
EPOCH 25 ...
Training Accuracy = 0.996
Validation Accuracy = 0.943
EPOCH 26 ...
Training Accuracy = 0.996
Validation Accuracy = 0.937
EPOCH 27 ...
Training Accuracy = 0.996
Validation Accuracy = 0.944
EPOCH 28 ...
Training Accuracy = 0.997
Validation Accuracy = 0.947
EPOCH 29 ...
Training Accuracy = 0.997
Validation Accuracy = 0.942
EPOCH 30 ...
Training Accuracy = 0.998
Validation Accuracy = 0.943
Model saved
```
* Finally model was tested on unseen test set. Following is the outcome of the same:
```
INFO:tensorflow:Restoring parameters from ./lenet
Train Accuracy = 0.998
Validation Accuracy = 0.943
Test Accuracy = 0.938
```
We don't see very large difference between Validation and test accuracy so which denotes model is stable.

## Test the Model on new images
I downloaded some of the images from the web for the classification. Following are the new images:

![33_turn_right_ahead.png](./CarND-Traffic-Sign-Classifier-Project\new_images\33_turn_right_ahead.png)
![35_Ahead_only.png](./CarND-Traffic-Sign-Classifier-Project\new_images\35_Ahead_only.png)
![38_keep_right.png](./CarND-Traffic-Sign-Classifier-Project\new_images\38_keep_right.png)
![39_keep_left.png](./CarND-Traffic-Sign-Classifier-Project\new_images\39_keep_left.png)
![40_roundabout_mandatory.png](./CarND-Traffic-Sign-Classifier-Project\new_images\40_roundabout_mandatory.png)

* New images were of different shape and were required to be converted to 32*32*3 shape. So function `read_resize_images(path, img_name)` was used to read and reshape the images.
* Then new images were preprocessed by using `preprocess()` function defined earlier.
* Then saved model was restored and was fit on the new images and new images were predicted. I got 100% accuracy on the new images.
* Finally top 5 softmax probabilities were plotted along with the image and it was found that probability of the correct classes were almost equal to 1 for all 5 test images. Following are the plot of the same:
![probability1.jpg](./CarND-Traffic-Sign-Classifier-Project/probability1.jpg)
![probability2.jpg](./CarND-Traffic-Sign-Classifier-Project/probability2.jpg)
![probability3.jpg](./CarND-Traffic-Sign-Classifier-Project/probability3.jpg)
![probability4.jpg](./CarND-Traffic-Sign-Classifier-Project/probability4.jpg)
![probability5.jpg](./CarND-Traffic-Sign-Classifier-Project/probability5.jpg)

## Further Improvements
Model is still overfitting the model. We can further improve this model by using following techniques:
* We can use histogram equalization so that the images which are very dark can be equalized so that traffic sign should be visible.
* We can augment and add some more data so that overfitting can be reduced.

For more information, have a look at Complete [project report](./Project_writeup.md) and [HTML file](./Traffic_Sign_Classifier.html) given in the repository.
