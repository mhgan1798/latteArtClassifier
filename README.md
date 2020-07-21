# Latte Art Classifier

## The model
This app uses a tf.keras convolutional neural network trained on images of latte art with a single-node dense layer to estimate the grade of an image of coffee (binary classification).

## The application
The app is build on the plotly-dash framework. The model is loaded in and used in app callbacks to carry out the classification when an image is uploaded.
![latteArtClassifier](https://user-images.githubusercontent.com/67821956/87681368-2b6e2900-c7b1-11ea-815a-77c1f4287682.gif)

## Grading scheme
Since the single-node linear-activation output of the neural network is inherently continuous, multiple outputs were generated from the test-set. Through qualitative quantile analysis of the distribution of the outputs, each output could be discretised into bins corresponding to the their respective grades (A+ to F).
