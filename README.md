# Gesture Recognizer
Recognizes sign language gestures: digits 0-9 and letters A-Z

## Authors
Reid Fu, Jiyuan Li, Siyuan Li

## Task Assignments
Project will consist of 3 main parts: preprocessing, feature extraction, and classification.
Preprocessing will be divided and conquered as follows:
- Noise removal (filtering algorithms): Siyuan Li
- Image resizing (scaling algorithms): Jiyuan Li
- Region segmentation: Reid Fu

We will use object skeleton, interest points, and principal component axes/values as features.
Feature extraction will be divided and conquered as follows:
- Principal component analysis: Reid Fu
- Skeletonization: Jiyuan Li
- Interest points: Siyuan Li

Classification will be done using a neural net, which will be done by Reid Fu

Data will be obtained from http://www.massey.ac.nz/~albarcza/gesture_dataset2012.html

## Description
How do humans recognize hand gestures? There are many variables in each gesture, including the color, size, and relative proportions of the gesturers' hands. However, the shape of the gesturer's hand is about the same for a given category of gesture. By this, we mean that the silhouette formed by a hand performing a certain gesture will be about the same no matter who's performing the gesture. An outstretched palm will be interpreted as a 5 no matter who's palm it is. The orientation of fingers inside the silhouette will also be about the same. For E, the non-thumb fingers will all be resting on the thumb, while for M, the thumb will be tucked between the ring and the little fingers.

To get the silhouette, we plan to use background subtraction. This method will generate a binary image in which pixels in the hand's silhouette will have values of 1, while the background pixels will have values of 0. From this binary image, we can get both a shape representation and a smaller window for edge/corner detection. We will use medial axis transform and principle component analysis to represent shape, and an ordered list of interest points to represent edges and corners.

The following information will be obtained from the training set:
- Mean and variance in medial axis transform for each gesture category
- Mean axes and variances in PCA eigenvectors for each gesture category
- The interest points detected by the FAST algorithm on segmented images

These features will be used to build a neural network. The network will then evaluate test images based on:
- The first 7 similitude moments of segmented images
- Euclidean distance between test image's PCA eigenvectors and mean PCA eigenvectors for each category
- The number of interest points detected by the FAST algorithm on segmented images
- All of above will be converted into distance in standard deviations

## Neural Network Architecture
