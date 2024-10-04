import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import os

# Define a custom L2Normalization layer to replace Lambda
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

# Define custom object for L2Normalization layer deserialization
custom_objects = {
    'L2Normalization': L2Normalization
}

# Load the trained model with the custom L2Normalization layer
model = load_model('recommendation_model.keras', custom_objects=custom_objects)

# Load customer and article features
customer_features_with_id = np.load('customer_features.npy', allow_pickle=True).item()
articles_features_with_id = np.load('articles_features.npy', allow_pickle=True).item()

# Load the customer dataset (adjust the path if necessary)
df_c = pd.read_csv("C:/Users/User/OneDrive/Desktop/PROJECT/customers.csv")

# Assuming df_a contains a column 'image_name' with the path or filename of images
df_a = pd.read_csv("C:/Users/User/OneDrive/Desktop/PROJECT/df_a_images.csv")
article_image_paths = dict(zip(df_a['article_id'], df_a['image_name']))

# Function to get recommendations
def get_recommendations(customer_id):
    customer_features = customer_features_with_id.get(customer_id, None)
    if customer_features is None:
        st.error("Customer ID not found!")
        return None, None

    article_ids = np.array(list(articles_features_with_id.keys()))
    article_features = np.array([articles_features_with_id[article_id] for article_id in article_ids])

    customer_features = np.tile(customer_features, (len(article_features), 1))
    scores = model.predict([customer_features, article_features])

    top_indices = np.argsort(-scores, axis=0)[:10].flatten()  # Top 10 recommendations
    recommended_articles = article_ids[top_indices]
    recommended_images = [article_image_paths.get(article_id, None) for article_id in recommended_articles]

    return recommended_articles, recommended_images

# Interface styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    body {
        font-family: 'Montserrat', sans-serif;
    }

    .title {
        font-family: 'Montserrat', sans-serif;
        font-size: 55px;
        color: black;
        text-align: center;
    }

    .header {
        font-family: 'Montserrat', sans-serif;
        font-size: 35px;
        color: black;
        text-align: center;
    }

    .subheader-member {
        font-family: 'Montserrat', sans-serif;
        font-size: 25px;
        color: black;
        text-align: center;
    }

    button {
        font-family: 'Montserrat', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Logo and title
logo_url = "https://upload.wikimedia.org/wikipedia/commons/5/53/H%26M-Logo.svg"
st.markdown(f"""
    <div style="text-align: center;">
        <img src="{logo_url}" alt="H&M Logo" style="width: 150px; height: auto; margin-right: 20px;">
        <h1 class="title" style="display: inline-block; font-size: 45px; vertical-align: middle;">Christmas Giveaway</h1>
    </div>
    """, unsafe_allow_html=True)

# Christmas-themed image
image_url = "https://parc-le-breos.co.uk/wp-content/uploads/2017/09/Christmas-Bow-PNG-Image.png"
st.image(image_url, caption=None, use_column_width=True)

# Header
st.markdown('<h2 class="header">"Experience more this Christmas"</h2>', unsafe_allow_html=True)

# Spacer
st.markdown("""
    <style>
    .space { margin-top: 50px; }
    </style>
    <div class="space"></div>
    """, unsafe_allow_html=True)

# Membership number input container (separate container)
with st.container():
    st.markdown('<h2 class="subheader-member">Are you an H&M member?</h2>', unsafe_allow_html=True)

    with st.form(key='member_form'):
        membership_number = st.text_input("If yes, please type your H&M membership number below:")

        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            if membership_number in df_c['customer_id'].astype(str).values:
                st.success("You are successfully signed in, please answer the following questions.")
            else:
                st.error("Unfortunately, the number is incorrect, please try again!")

# Non-member button
st.markdown("""
    <div class="button-container" style="text-align: center; margin-top: 20px;">
        <a href="https://www2.hm.com/en_gb/member/info.html" target="_blank">
            <button class="btn-red" style="background-color: #f63366; color: white; padding: 10px 20px; border: none; border-radius: 5px;">Not yet, I want to join!</button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Spacer
st.markdown('<div class="space"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<h2 class="subheader-small">Please answer the following questions to enter the competition:</h2>', unsafe_allow_html=True)

# Questions
st.slider("How satisfied are you with the overall shopping experience at H&M?", 1, 10, 10)
st.slider("How satisfied are you with the quality and variety of H&M products?", 1, 10, 10)
st.slider("How would you rate the customer service you receive at H&M?", 1, 10, 10)
st.checkbox("By clicking here, I state that I have read and understood the terms and conditions*.")

# Submit button
if st.button("Submit"):
    st.markdown("""
        <div style="text-align: center; font-size: 20px; font-family: 'Montserrat', sans-serif;">
            <strong>We are grateful to our members for spending another year with us.</strong><br>
            <strong>As a token of our appreciation, we would like to gift a few of you with an item of your choice.</strong><br>
        </div>
        """, unsafe_allow_html=True)

    # Display Christmas-themed image
    image_url = "https://imageio.forbes.com/specials-images/imageserve/5fb5c024bf3bf4129d04e891/Red-and-Green-Gift/960x0.jpg?format=jpg&width=1440"
    st.image(image_url, caption=None, use_column_width=True)

    st.markdown("""
        <div style="text-align: center; font-size: 20px; font-family: 'Montserrat', sans-serif;">
            <strong>Below are the items we have personally selected for you.</strong><br>
            <strong>Choose your favorite, and if you're lucky, it will be under your Christmas tree.</strong><br>
        </div>
        """, unsafe_allow_html=True)

    # Add more spacing
    st.markdown("""
        <style>
        .space {
            margin-top: 50px;
        }
        </style>
        <div class="space"></div>
        """, unsafe_allow_html=True)

    # Display recommended items after the Montserrat content
    recommendations, recommended_images = get_recommendations(membership_number)
    if recommendations is not None and recommended_images is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container():
                if recommended_images[0] and os.path.exists(recommended_images[0]):
                    st.image(recommended_images[0], caption=f"Article ID: {recommendations[0]}", use_column_width=True)
                else:
                    st.write(f"Article ID: {recommendations[0]} - Image not found")
            st.checkbox("Select this item", key="checkbox1")

        with col2:
            with st.container():
                if recommended_images[1] and os.path.exists(recommended_images[1]):
                    st.image(recommended_images[1], caption=f"Article ID: {recommendations[1]}", use_column_width=True)
                else:
                    st.write(f"Article ID: {recommendations[1]} - Image not found")
            st.checkbox("Select this item", key="checkbox2")

        with col3:
            with st.container():
                if recommended_images[2] and os.path.exists(recommended_images[2]):
                    st.image(recommended_images[2], caption=f"Article ID: {recommendations[2]}", use_column_width=True)
                else:
                    st.write(f"Article ID: {recommendations[2]} - Image not found")
            st.checkbox("Select this item", key="checkbox3")

    st.markdown("""
        <div style="text-align: center; font-size: 20px; font-family: 'Montserrat', sans-serif; color: red;">
            The winners will be announced on December 12th.
        </div>
    """, unsafe_allow_html=True)

# Additional space for the UI
st.markdown('<div class="space"></div>', unsafe_allow_html=True)
