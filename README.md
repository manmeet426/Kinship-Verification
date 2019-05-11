# Kinship-Verification
Neural network project

How to run the code for training and testing the neural network.

The code repository can be found at the following github link.
Github Link : 
https://github.com/manmeet426/Kinship-Verification

The datasets KinFaceW-I and KinFaceW-II are located in the file system and contain their own separate readme files.
The main code for the neural network as well as the GUI is contained in NN_project.ipynb
The load data function loads the files(images) from all the classes (f-d, f-s, m-d, m-s), prepares labels for the same and calls import_dir which performs the function to merge the parent and child images into a single image as required for input to the model.
It returns all images from all folders into parent-child image form.


Import_dir function reads all files(images) for a given directory (say father-dau) and combines parent-child images to form a single 6 channel image which is used as input for the proposed model. The start variable is required to skip any thumbs.db file that may be present in the folder containing the images. It return parent-child images for the given directory.

Now, using the KinFaceW-II dataset, we load the train data and labels in the input format required by the model (parent-child combination 6-channel image).
Further, we combine the train_data from different class folders (father-daughter, father-son, mother-daughter, mother-son) into a single variable to pass to the input of the neural net.

We do a similar procedure for the test data and test labels using the KinFaceW-I dataset.

We then convert the labels into categorical form for the multi-class classification task.

Now, for the proposed model, we prepare the data generator for data augmentation.

We then fit the model with the augmented train data to get the training accuracy as stated above. 
We evaluate the performance of the  model using the accuracy on the test data which is shown above.

 the above model is saved as ‘model_with_data_augmentation’ in the same root file structure.

model.save("model_with_data_augmentation")
	
The model can be re-loaded using the following code and run again to calculate the accuracy.

models.load_model("model_with_data_augmentation")




How to use the GUI?

The NN_project.ipynb file contains the code for the GUI at the end. It describes the function performed by the GUI.

The 2 upload buttons allow the user to upload input images for the classification task. Once the user clicks submit, the GUI code transforms the uploaded images into the parent-child format required for the model input.
The code calls the model’s predict classes function to obtain the prediction for the uploaded images.
The resulting class is displayed in a separate dialog box.

The following video shows a demonstration on how to use the GUI for uploading the input images and viewing the result.


Link to Demo video on Youtube : https://www.youtube.com/watch?v=W4XAuGpUA_M&feature=youtu.be

Link to github webpage : https://github.com/manmeet426/Kinship-Verification

Kinship Verification via facial images is an emerging research topic in the field of biometrics, pattern recognition and computer vision. It is motivated by the findings that individuals with some genetic relations have certain similarities in their facial appearances. These similarities in the facial appearance is a result of inherited facial features from one generation to the next generation, especially from parents to children.
Approach: 
For performing kinship verification we deploy a convoluted neural network which takes a pair of images as input and outputs the probability of these images belonging to one of four classes given by father-daughter, father-son, mother-daughter and mother-son. 
For achieving this, we use the Kinship Face in the Wild dataset (http://www.kinfacew.com/), which is a database of face images collected for studying the problem of kinship verification from unconstrained face images.
The dataset contains two parts KinFaceW-I and KinFaceW-II. KinFaceW-I contains 268 images for the father-daughter class (134 father images, 134 daughter images), 312 images for the father-son class, 254 images for the mother-daughter class and 232 images for the mother-son class. That is a total of 533 pairs of images. KinFaceW-II contains 500 images for the father-daughter class (250 father images, 250 daughter images), 500 images for the father-son class, 500 images for the mother-daughter class and 500 images for the mother-son class. That is a total of 1000 pairs of images. Thus, for our model, we use the KinFaceW-II dataset for training the neural network and KinFaceW-I dataset for testing the accuracy of the network.

The approach for developing the neural network is as follows. First, we load the images in form of parent-child pairs and combine them into a single 6-channel image. We then pass this (64x64 pixel ) 6-channel image as input to the neural network with the architecture described below. 
model = models.Sequential()
    model.add(layers.Conv2D(16, (5, 5), strides=1, padding='valid', activation='relu', input_shape=(64, 64, 6)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


The network takes the input and passes it to a conv2D layer with a filter of size 5x5, which produces output of size (60, 60, 	16). The other input and output dimensions for the remaining layers are as follows:

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 60, 60, 16)        2416      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 30, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 26, 26, 32)        12832     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 9, 64)          51264     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 3, 64)          16448     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               33280     
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 2052      
=================================================================
Total params: 118,292
Trainable params: 118,292
Non-trainable params: 0

After the last Max-Pooling layer the output is flattened using a flatten layer. This flatten output is passed through a dense layer with 512 outputs. The output of which is passed through a softmax layer which classifies the input into 1 of 4 classes.  

The activation function used for each of the conv2d layers is Rectified Linear Unit (ReLU). Various other variations for ReLU were tried, but ReLU resulted in the best performance. 

The optimizer used is Stochastic Gradient descent with a learning rate of 0.01 and other hyper-parameters as shown in the network architecture.

The loss function used is categorical cross entropy since we need to perform multi-class classification. The evaluation metric for the model is set to its accuracy.

Initially, the performance obtained using the above network was close to 45-46% training accuracy and 33-34% validation accuracy. 

Hence, I implemented data augmentation to increase the size of the available data samples for the network to train on.

The code for Data Augmentation is as follows.

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.0,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=True) 

datagen.fit(train_img)


Now, we fit the model to train on the augmented data set as follows.

model.fit_generator(datagen.flow(train_img/255, train_labels_cat), steps_per_epoch=train_img.shape[0], epochs=10, verbose=1, shuffle=True)   

Here, train_img contains the training images formed by combining the images into parent-child pairs (6-channel images), train_labels_cat contains the labels for each image obtained after using the to_categorical function from keras.utils.

The accuracy obtained on the training set (KinFaceW-II) are as follows.

Epoch 1/10
1000/1000 [==============================] - 281s 281ms/step - loss: 1.2324 - acc: 0.3960
Epoch 2/10
1000/1000 [==============================] - 287s 287ms/step - loss: 0.9203 - acc: 0.5629
Epoch 3/10
1000/1000 [==============================] - 279s 279ms/step - loss: 0.7240 - acc: 0.6733
Epoch 4/10
1000/1000 [==============================] - 275s 275ms/step - loss: 0.5457 - acc: 0.7694
Epoch 5/10
1000/1000 [==============================] - 285s 285ms/step - loss: 0.4022 - acc: 0.8383
Epoch 6/10
1000/1000 [==============================] - 287s 287ms/step - loss: 0.3126 - acc: 0.8790
Epoch 7/10
1000/1000 [==============================] - 283s 283ms/step - loss: 0.2774 - acc: 0.8974
Epoch 8/10
1000/1000 [==============================] - 287s 287ms/step - loss: 0.2293 - acc: 0.9179
Epoch 9/10
1000/1000 [==============================] - 275s 275ms/step - loss: 0.2091 - acc: 0.9260
Epoch 10/10
1000/1000 [==============================] - 283s 283ms/step - loss: 0.1692 - acc: 0.9413


The accuracy obtained on the test set (KinFaceW-I) is obtained by

i/p : model.evaluate(test_img/255, test_labels_cat, batch_size=15)

o/p:     533/533 [==============================] - 2s 4ms/step

[2.661236596319957, 0.4915572347549292]

Thus, the accuracy obtained from this model is 49.15%.


Improvements: 

In order to improve the performance accuracy, I performed the following changes.

Added Batch normalization for each layer.
Experimented with different activation functions for each layer. 

After doing these modifications, the accuracy obtained on the training set (KinFaceW-II) are as follows.

Epoch 1/10
1000/1000 [==============================] - 965s 965ms/step - loss: 0.4201 - acc: 0.8032
Epoch 2/10
1000/1000 [==============================] - 1541s 2s/step - loss: 0.2669 - acc: 0.8818
Epoch 3/10
1000/1000 [==============================] - 533s 533ms/step - loss: 0.1725 - acc: 0.9291
Epoch 4/10
1000/1000 [==============================] - 533s 533ms/step - loss: 0.1168 - acc: 0.9539
Epoch 5/10
1000/1000 [==============================] - 584s 584ms/step - loss: 0.0901 - acc: 0.9661
Epoch 6/10
1000/1000 [==============================] - 520s 520ms/step - loss: 0.0697 - acc: 0.9740
Epoch 7/10
1000/1000 [==============================] - 529s 529ms/step - loss: 0.0588 - acc: 0.9792
Epoch 8/10
1000/1000 [==============================] - 5059s 5s/step - loss: 0.0568 - acc: 0.9793
Epoch 9/10
1000/1000 [==============================] - 840s 840ms/step - loss: 0.0381 - acc: 0.9862
Epoch 10/10
1000/1000 [==============================] - 538s 538ms/step - loss: 0.0375 - acc: 0.9872


The new accuracy obtained on the test set (KinFaceW-I) is obtained by

i/p : model.evaluate(test_img/255, test_labels_cat, batch_size=15)

o/p:     533/533 [==============================] - 4s 7ms/step

[1.141733029993569, 0.7528142565634193]


Thus, the accuracy improved to 75.28% which is comparable with the baseline accuracy as stated in http://www.bmva.org/bmvc/2015/papers/paper148/paper148.pdf.


