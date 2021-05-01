### Script for CS329s ML Deployment Lec 
import os
import json
import numpy as np
import requests
import SessionState
import streamlit as st
import tensorflow as tf
from prediction import text_vectorization, predict_json

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "acoustic-skein-309118-57d660baa292.json" # change for your GCP key
PROJECT = "acoustic-skein-309118" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)
MODEL = "amazon_review_model" # your model name


@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(text, model):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    
    text_vector = text_vectorization(text)
    print(text_vector)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    #text_vector = tf.cast(tf.expand_dims(text_vector, axis=0), tf.int16)
    #text_vector = tf.expand_dims(text_vector, axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=text_vector)
    print(preds)
    pred_class = np.array(preds)
    if pred_class>=0.3:
        out_label = 'Positive'
    else:
        out_label = 'negative'
    #print(pred_class)
    return out_label



### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to Sentiment analysis for product reviews")
st.header("Classify if the review is positive or negative!")
st.markdown('The model training and prediction code is covered in the corresponding GitHub repository https://github.com/yashinaniya/Amazon-review-classification-using-Tensorflow.')
text_input = st.text_area('Text:')


# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
#session_state = SessionState.get(pred_button=False)
#pred_button = st.button("Predict")

# Create logic for app flow
# if not uploaded_file:
    # st.warning("Please upload an image.")
    # st.stop()
# else:
    # session_state.uploaded_image = uploaded_file.read()
    # st.image(session_state.uploaded_image, use_column_width=True)
    # pred_button = st.button("Predict")

# Did the user press the predict button?
#if pred_button:
    #session_state.pred_button = True 

# And if they did...
#if session_state.pred_button:
if st.button("Predict"):
    pre = make_prediction(text_input, model=MODEL)
    st.write(f"Prediction: {pre}")

 

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()