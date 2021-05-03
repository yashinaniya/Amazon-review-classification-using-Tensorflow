# Amazon-review-classification-using-Tensorflow
Sentiment analysis of large amazon review dataset using a deep learning neural network build using LSTM layers 
In this repo we use the Mobiel_Electronics dataset(subset of the amazon review dataset).
Although I have achieved high accuracy on both training and test set, but feel free to play with hyperparameters to improve it further.
Also, you can add more LSTM layers or instead use RNN or CNN for building the neural network.

Along with the neural network model, I also tried simple Classifier like [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) and [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
There are other classifiers like Random forest and SVM which should give better accuracy. Feel free to try it.

## What is needed to develop this application?
 - A [Google Cloud Account](https://cloud.google.com/gcp) and [Cloud Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
 - A [Google Cloud SDK Utility](https://cloud.google.com/sdk/docs/install)
 - A [Docker container](https://docs.docker.com/get-docker/)
 Please note: Using Google cloud services will incur charges. If you are creating a new account, you will be provided with 300 US$ credit, otherwise you'll pay for the services you use.
 

