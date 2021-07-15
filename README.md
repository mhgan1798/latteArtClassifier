# Latte Art Classifier
http://latte-art-classifier.herokuapp.com/

## The model
This app uses a tf.keras convolutional neural network trained on images of latte art with a single-node dense layer to estimate the grade of an image of coffee (binary classification).

## The application
The app is built on the plotly-dash framework. The model is loaded in and used in app callbacks to carry out the classification when an image is uploaded.

![latteArtClassifier](https://user-images.githubusercontent.com/67821956/87681368-2b6e2900-c7b1-11ea-815a-77c1f4287682.gif)

## Grading scheme
Given that the single-node linear-activation output of the neural network produces a continuous value, the distribution of predictions generated from the entire data-set can be used to determine a grading scheme. In other words, through quantile analysis of the distribution of the predictions of the testing and training sets, each output could be discretised into qualitatively-determined bins corresponding to the their respective grades (A+ to F).
