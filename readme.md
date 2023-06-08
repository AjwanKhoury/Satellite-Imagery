# Satellite Imagery

![GitHub stars](https://img.shields.io/github/stars/AjwanKhoury/Satellite-Imagery?style=social) ![GitHub last commit](https://img.shields.io/github/last-commit/AjwanKhoury/Satellite-Imagery)

## Goals:
Development of an image classification system for a satellite, including the ability to identify horizons, star patterns, and flashes, the ability to identify "good images," image compression, and adaptation for transmission. In the framework of the project, we aim to achieve the following capabilities:

1. Given satellite images, we want to classify them into the following categories:
   1.1. Images of stars.
   1.2. Images of Earth during the day.
   1.3. Images of Earth at night with significant city lights.
   1.4. Images that capture both Earth and the space, clearly showing the curvature of the Earth.
   1.5. Invalid images that should be classified as irrelevant.

2. Given a collection of images and a parameter and category, we want to sort the images based on their relevance (the most beautiful and suitable at the top, and so on).

3. Given an image, we want to identify and extract the most "interesting" region. For example, if it's an image of Earth, we want to extract the most sharply defined area (e.g., land rather than sea).

4. Aggressive compression of images due to limited satellite communication resources. Using image processing techniques, we want to compress the images to a size of 10-20kB (VGA size).

## General Description and Workflow
As part of the topic learning and article reading, we decided to choose a deep learning model of the CNN type to perform image classification. This model has shown the best results in all the research conducted on image recognition, so we chose it. In the first stage, we searched for online images that we could use for training the model. After finding enough images that cover all the capabilities we want to implement, we started writing the code and planning the project. We created a user interface using the PyQt5 library in Python, which allows us to retrain the model or make predictions on input images. Currently, the user enters the input, but in the future, we want to connect it directly to the satellite. Additionally, we did the same for the capability to perform image segmentation and identify interesting parts. We decided that water sources are the interesting parts in the image, so we can train the segmentation model through the interface or load an image and receive a cutout of the part containing the water source. The percentage that it represents from the total image will also be displayed. Another option is sorting images based on our level of interest. Since water sources are what interests us, the code will sort all the loaded images based on the amount of water present in them, from small to large, and provide an option to navigate between them. Finally, we created an image compression capability that can reduce each image to a size of 10KB-20KB suitable for satellite transmission.

For the first capability of image classification, we implemented it using a CNN deep learning model, which allows classifying images into different categories. First, we prepared the data and translated it into a format that the model can work with. We created a Python DataFrame that contains the image locations and their labels, and finally saved it in a CSV file. In the next stage, we processed the images using the ImageDataGenerator function from the Keras library. The processing included pixel value normalization, augmentation, and conversion of images and labels into tensors. Finally, we split the data into training and validation sets. In the next step, we built a CNN model with several dense convolutional layers and a softmax layer. Finally, the model was compiled using the Adam optimizer, a loss function based on categorical cross-entropy, and accuracy as the evaluation metric.

In the next step, we trained the model using 50 epochs and then performed validation on the test data. Finally, we saved the model in a file for future preservation. This way, we don't have to train the model from scratch each time and can directly load it on the satellite's task computer without wasting valuable resources. We created a function called "predict" that performs prediction on new images. It takes an image as input, loads it into the trained model, and returns the probabilities for each class (label), selecting the class with the highest probability as the predicted class.

The second and third capabilities of performing image segmentation and object cropping of interesting parts from the input image, as well as sorting images based on their level of interest, were accomplished using a U-Net model for image segmentation. We created contraction and expansion functions for the image, which served as layers in our model. During the model construction, we combined the features obtained from the contraction and expansion functions to help the model preserve important information. 

Next, we prepared the dataset and split it into training and testing data, compiled the model, and trained it. We also saved the trained model to avoid retraining on the same data. Similarly, we created a prediction function that takes an image as input, processes it, and returns the most important parts of the image according to our defined criteria (identification of water segments).

Additionally, we developed a sorting function that takes a list of images, identifies the interesting regions in each image, and sorts them based on the percentage of the interesting area within the entire image.

To implement the final capability of compressing images to a size of 10-20KB, we created a function that takes an image as input, checks its original size, defines the desired quality of the compressed image, and performs compression only if the loaded image is larger than 20KB. We then convert the image to RGB mode and save the new image with the calculated quality while optimizing it. This way, we have implemented all the capabilities defined in the project objectives.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/AjwanKhoury/Satellite-Imagery
   ```

2. Navigate to the project directory:

   ```
   cd Satellite-Imagery
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Results:

For the initial capability of image classification, we were able to achieve an accuracy of over 90% for the model we built. This is a high level of accuracy that satisfies us relative to the computational limitations of the task's computer. Of course, we can further improve this in the future by adding additional training images and expanding the model to larger datasets. \
_Performing classification, for example, on an uploaded image that challenges the model:_\
\
![image](https://github.com/AjwanKhoury/Satellite-Imagery/assets/73795045/91a2f024-8682-4b08-a54d-7b0de0080d33) \
\
It can be seen that the model successfully classified the image as a picture of planet Earth with illuminated cities. Of course, there are other categories that the model can classify, such as images of stars, images of the Earth itself in space, and images that specifically show the Earth/ocean without a space background. \
In the second and third capabilities, which deal with identifying interesting areas in the image and ranking them based on their level of interest, we were able to achieve over 90% accuracy. The tool is able to take an image and mark the water areas appearing in the image, while calculating the percentage of water within the entire image.\
\
_Classification example:_ \
\
![image](https://github.com/AjwanKhoury/Satellite-Imagery/assets/73795045/cf59e67d-4965-463c-bacb-92e9276472df) \
\
The sorting capabilities also work excellently. After running the sorting function, we received a sorting of images from the lowest percentage of water sources to the highest percentage. An example of the first three images: \
First image - Ranking 0, 16.65% water from the entire image. \
\
![image](https://github.com/AjwanKhoury/Satellite-Imagery/assets/73795045/742fd011-b02d-4fa4-b6f6-007a3810ea92) \
Image 2 - Ranking 1 - 17.24% water from the entire image.
\
Image 3 - Ranking 2 - 17.6% water from the entire image. \
\
![image](https://github.com/AjwanKhoury/Satellite-Imagery/assets/73795045/6e284f2e-20cf-4c13-9e78-7d8913e3058a) \
\
The last capability of image compression was successfully implemented for every loaded image. \
_An example of image compression:_ \
\
![image](https://github.com/AjwanKhoury/Satellite-Imagery/assets/73795045/d7f8872b-a13c-4d6f-9dc8-3bbd34eaff8b) \
\
On the left is the original image, and on the right is the compressed image. Overall, we were able to reduce the image by 62.19% of its original size, bringing it down to a total size of 14.16KB without compromising its quality. It's difficult to distinguish between the image on the left (not compressed) and the compressed image on the right. 

## Findings
Using image processing techniques and deep learning, we have successfully implemented a useful tool that can handle satellite images, classify them into classes, extract important regions, and compress them. Throughout the project, we learned new technologies and capabilities that can be highly useful for specific industries that utilize imaging. We discovered that with deep learning, it is possible to relatively accurately catalog images if the model is well-defined and sufficient training data is available. Additionally, we found that it is possible to compress images while preserving their quality to some extent. Furthermore, we learned how to extract interesting parts from the images. In our project, we decided that a water source is an interesting area, but we could have also chosen that grasslands, fields, or desert and sandy areas are interesting regions and perform the cutting accordingly. There are numerous extensions and applications that can be performed on the current project, as it is a fascinating field with almost endless possibilities in almost any aspect of our lives.


## Contact

For any inquiries or questions, please contact [ajwan_khoury2000@hotmail.com](mailto:ajwan_khoury2000@hotmail.com).

---
