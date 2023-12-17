from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib notebook
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
from skimage.transform import resize
import skimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint 
from sklearn.preprocessing import LabelEncoder
import itertools

dimension=(300, 300)
num_epochs = 500
num_batch_size = 32
num_channels = 3
model = Sequential()


def preprocess_image(file):
    img = skimage.io.imread(file)
    img =  resize(img, dimension, anti_aliasing=True, mode='reflect')
#    img = np.array(img)
    return img
    

#Load images in structured directory like it's sklearn sample dataset
def load_image_files(container_path):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    images = []
    target = []
    for i, direc in tqdm(enumerate(folders)):
        for file in tqdm(direc.iterdir()):
#            img = skimage.io.imread(file)
            img_resized = preprocess_image(file)
            images.append(img_resized)
            target.append(categories[i])
            
    target = np.array(target)
    images = np.array(images)

    return Bunch(target=target,
                 target_names=categories,
                     images=images)
    

    
image_dataset = load_image_files('./dataset/')
#image_dataset = load_image_files("E:\Dnew\Ff\Thesis\Main Things\Mosquito Classification\dataset")

#Encode all level
le = LabelEncoder()
yy = to_categorical(le.fit_transform(image_dataset.target)) 
num_labels = yy.shape[1]

#Split data
X_train, X_test, y_train, y_test = train_test_split(image_dataset.images, yy, test_size=0.3,random_state=109)



def construct_model():
    # Construct model     
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=dimension+tuple([num_channels]), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(num_labels, activation='softmax'))


def compile_model():
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    

def model_summary():
    # Display model architecture summary 
    model.summary()
    
    # Calculate pre-training accuracy 
    score = model.evaluate(X_test, y_test, verbose=1)
    accuracy = 100*score[1]
    
    print("Pre-training accuracy: %.4f%%" % accuracy)


def train_model():
    # train the model
    checkpointer = ModelCheckpoint(filepath='./saved_model.h5/', 
                                   verbose=1, save_best_only=True)
    
    hist = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
    


    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    xc = range(num_epochs)

    plt.figure(1, figsize = (7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('Number of epochs')
    plt.ylabel('accuracy')
    plt.title("train acc vs val acc")
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.style.use(['classic'])

    plt.figure(2, figsize = (7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('Number of epochs')
    plt.ylabel('loss')
    plt.title("train loss vs val loss")
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.style.use(['classic'])
    

def evaluate_model():
    # Evaluating the model on the training and testing set
    score = model.evaluate(X_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])
    

def confussion_matrix():
    # confussion matrix
#    predictions = model.predict(x_test, batch_size=num_batch_size, verbose=0)
    class_predictions = model.predict_classes(X_test, batch_size=num_batch_size, verbose=0)
    class_names = image_dataset.target_names
    print(classification_report(np.argmax(y_test, axis=1), class_predictions, target_names = class_names))
    
    cm = confusion_matrix(np.argmax(y_test, axis=1), class_predictions)
#    for i in range(0, len(cm)):
#      print(cm[i])
      
    
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Reds)
    plt.title("Confussion Matrix")
    # plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation = 45)
    # plt.yticks(tick_marks, class_names)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j], 
               horizontalalignment = 'center',
               color = 'white' if i==j else 'black')
      
    
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    


def print_prediction(file_name):
    prediction_feature = preprocess_image(file_name) 
    prediction_feature = prediction_feature.reshape(((1,)+dimension+tuple([num_channels])))

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 
    



construct_model()
compile_model()
model_summary()
train_model()
evaluate_model()
confussion_matrix()
print_prediction('testimage.jpg')

    