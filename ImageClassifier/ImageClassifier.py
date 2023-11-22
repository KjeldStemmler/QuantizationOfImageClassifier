import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp

#method for unpacking the cifar Data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#method for quantizing a numpy array
def quantisation(inputArray, bitSize, fractBits = 0):
     #compute scale and zeroPoint
    maximum = np.max(inputArray);
    minimum = np.min(inputArray);
    valueRange = maximum - minimum;
    
    #using a fxpMath dummy to get the bounds for quantisation
    fxDummy = Fxp(0, (minimum < 0), bitSize, fractBits)
    maxFpVal = fxDummy.upper
    minFpVal = fxDummy.lower

    #scale = 1 when only one value is present to avoid division by 0
    scale = 1;
    if (valueRange != 0):
        scale = (maxFpVal - minFpVal) / valueRange;
    zeroPoint = minFpVal-(scale*minimum);
    
    #the actual quantisation and restructuring into a fxp-array
    output = Fxp((inputArray * scale) - zeroPoint)

    return output;

#import the first cifar dataset
batch1 = unpickle("cifarData/data_batch_1")
trainingImagesLinear = batch1.get(list(batch1.keys())[2])
trainingLabels = np.array(batch1.get(list(batch1.keys())[1]))


#import validation dataset
validationData = unpickle("cifarData/test_batch")
testImagesLinear = validationData.get(list(validationData.keys())[2])
testLabels = np.array(validationData.get(list(validationData.keys())[1]))

#reshape each picture from 1000, 3072 to 1000, 32, 32, 3
for i in range(10000):
    #separating the channels and reshaping into the 32-pixel-format
    pxls_R = trainingImagesLinear[i][0:1024].reshape(32,32)
    pxls_G = trainingImagesLinear[i][1024:2048].reshape(32,32)
    pxls_B = trainingImagesLinear[i][2048:3072].reshape(32,32)
    #reassembling the array usind depth-first-stack
    img = np.dstack((pxls_R, pxls_G, pxls_B))
    if (i == 0):
        #initializing the array-variable
        trainingImages = np.reshape(img, (1,32,32,3))
    else:
        trainingImages = np.vstack((trainingImages, np.reshape(img, (1,32,32,3))))

#repeating for the validation data
for i in range(10000):
    pxls_R = testImagesLinear[i][0:1024].reshape(32,32)
    pxls_G = testImagesLinear[i][1024:2048].reshape(32,32)
    pxls_B = testImagesLinear[i][2048:3072].reshape(32,32)
    img = np.dstack((pxls_R, pxls_G, pxls_B))
    if (i == 0):
        testImages = np.reshape(img, (1,32,32,3))
    else:
        testImages = np.vstack((testImages, np.reshape(img, (1,32,32,3))))


#set to True to display some of the images for inspection
if (False):
    plt.figure(figsize=(10,10))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(testImages[i])
        plt.xlabel(class_names[trainingLabels[i]])
    plt.show()


#convolution layers for feature extraction
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#fully connected layers for class prediction
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#print the current model summary
model.summary()

#preparing model for training
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#training the model
history = model.fit(trainingImages, trainingLabels, epochs=10, 
                    validation_data=(testImages, testLabels))

#set to True to display the changing accuracy of the model during training
if(False):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

#display the accuracy of the trained model
test_loss, test_acc = model.evaluate(testImages,  testLabels, verbose=2)
print(test_acc)

#get the weights of the model 
weights = model.get_weights()

#quantizing the weights with differently sized integers and FP-ranges, then storing th
accuracys = {}  
#iterating over different amounts of fractional bits
for fractBits in [0, 2, 4, 6, 8]:
    accuracys.update({"FP " + str(fractBits) : [0]})
    #iterating over a range of sizes
    for bits in range(1, 17):

        #resetting the container for quantized weights
        quantizedWeights = []
        #quantizing each element in the weights-list
        for i in range(len(weights)):
            quantizedWeights.append(quantisation(np.array(weights[i]), bits, min(fractBits, bits)))
        #update the model with the new weights
        model.set_weights(quantizedWeights)
        #test the accuracy of the updated model and save the accuracy for display
        test_loss, test_acc = model.evaluate(testImages,  testLabels, verbose=2)
        accuracys["FP " + str(fractBits)].append(test_acc)
        
#display, how accurate each quantization was
plt.xlabel("Number of bits used")
plt.ylabel("Accuracy")
plt.plot(accuracys["FP 0"], label="Integers")
plt.plot(accuracys["FP 2"], label="FP 2")
plt.plot(accuracys["FP 4"], label="FP 4")
plt.plot(accuracys["FP 6"], label="FP 6")
plt.plot(accuracys["FP 8"], label="FP 8")
plt.legend()
plt.show()

    