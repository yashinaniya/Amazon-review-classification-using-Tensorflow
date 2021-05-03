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
 
## The end Product

If you follow the steps below, you would be able to end up with a [Streamlit](https://streamlit.io/) web application.

##Getting the app running
1. Clone this repo
```
git clone 
```
2. Change into the Prediction directory
```
cd Prediction
```
3. Create and activate a virtual environment
```
pip install virtualenv
virtualenv <ENV-NAME>
.\<ENV-NAME>\Scripts\activate
```
4. Install the required dependencies (Streamlit, TensorFlow, etc)
```
pip install -r requirements.txt
Activate Streamlit and run app.py
streamlit run app.py
```
## To run the application successfully, you will need to make the following changes:

1. Create a [Project](https://cloud.google.com/appengine/docs/standard/nodejs/building-app/creating-project) on GCP
2. Create a [Bucket](https://cloud.google.com/storage/docs/creating-buckets) where you will store your Tensorflow SavedModel.
3. Upload your savedmodel directly to the bucket from the Google colab 
```
## Uploading a model to Google Storage from within Colab ##

# Authorize Colab and initalize gcloud (enter the appropriate inputs when asked)
from google.colab import auth
auth.authenticate_user()
!curl https://sdk.cloud.google.com | bash
!gcloud init

# Upload SavedModel to Google Storage Bucket
!gsutil cp -r <YOUR_MODEL_PATH> <YOUR_GOOGLE_STORAGE_BUCKET>
```
4. [Connect your model in bucket to the AI Platform](https://cloud.google.com/ai-platform/prediction/docs/deploying-models). You can do this step either manually through the GCP console or through gcloud Command line interface.
5. Create a [Service account](https://cloud.google.com/iam/docs/creating-managing-service-accounts) and then a version for your model.
6. Once we have concluded all these steps, we can generate a new JSON KEY from the saved model version. This key needs to be updated in the app_classification.py file.
7. Update the following variables:
 - In the app_classification.py, make the following changes:
```
# Google Cloud Services look for these when your app runs

# Old
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "acoustic-skein-309118-57d660baa292.json"

# New 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<PATH_TO_YOUR_KEY>"
```
- Update the Project name, region and Model
```
# Old
PROJECT = "acoustic-skein-309118"
REGION = "us-central1" 
MODEL = "amazon_review_model"
# New
PROJECT = "<YOUR_GCP_PROJECT_NAME>"
REGION = "<YOUR_GCP_REGION>"
MODEL = "<YOUR_MODEL_NAME>"
```
8. Deploy the whole app in GCP
 - run `make gcloud-deploy` command from your activated virtual environment


### if you face any problems relating to working of the application or would like help in developing a similar app for different application feel free to connect with me at [Linkedin](https://www.linkedin.com/in/yash-inaniya-558571bb/)
