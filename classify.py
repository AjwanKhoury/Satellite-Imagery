import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def build_model():
    class_names = ['Earth', 'Earth with space', 'Earth with light city', 'Star']
    # Create an empty dataframe
    data = pd.DataFrame(columns=['image_path', 'label'])

    # Define the labels/classes
    labels = {'data/Classify/{}'.format(class_names[0]) : class_names[0],
            'data/Classify/{}'.format(class_names[1]) : class_names[1],
            'data/Classify/{}'.format(class_names[2]) : class_names[2],
            'data/Classify/{}'.format(class_names[3]) : class_names[3]}

    # Loop over the train, test, and val folders and extract the image path and label
    for folder in labels:
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            label = labels[folder]
            data.loc[len(data.index)] = [image_path, label] 
            
    # Save the data to a CSV file
    data.to_csv('image_dataset.csv', index=False)

    # Load the dataset
    df = pd.read_csv("image_dataset.csv")

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Pre-process the data
    train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    rotation_range=45,
                                    vertical_flip=True,
                                    fill_mode='nearest')


    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                        x_col="image_path",
                                                        y_col="label",
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode="categorical")

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                    x_col="image_path",
                                                    y_col="label",
                                                    target_size=(64, 64),
                                                    batch_size=16,
                                                    class_mode="categorical")

    # Build a deep learning model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=50, validation_data=test_generator)

    # save it as a h5 file

    from tensorflow.keras.models import load_model

    model.save('model/classification_mdl.h5')
    
def predict(mdl, imgPath):
    # Define the class names
    class_names = ['Earth with light city',  'Earth', 'Earth with space', 'Star']
    
    # Load an image from the test set
    img = load_img(imgPath, target_size=(64, 64))

    # Convert the image to an array
    img_array = img_to_array(img)

    # Normalize the image pixel values to the range [0, 1]
    img_array = img_array / 255.0
    img_array = np.reshape(img_array, (1, 64, 64, 3))

    # Get the model predictions
    predictions = mdl.predict(img_array)

    # Get the class index with the highest predicted probability
    class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_label = class_names[class_index]
    print(class_index)
    return predicted_label

    