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
How do humans recognize hand gestures? There are many variables in each gesture, including the color, size, and relative proportions of the gesturers' hands. However, the shape of the gesturer's hand is about the same for a given category of gesture. By this, we mean that the silhouette formed by a hand performing a certain gesture will be about the same no matter who's performing the gesture. An outstretched palm will be interpreted as a 5 no matter who's palm it is.
The orientation of fingers inside the silhouette will also be about the same. For E, the non-thumb fingers will all be resting on the thumb, while for M, the thumb will be tucked between the ring and the little fingers.

To represent silhouette shape, we plan to use ?????
