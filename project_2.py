#!/usr/bin/env python
# coding: utf-8

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import nltk#!/usr/bin/env python
# coding: utf-8

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Dot, Layer  # Import Layer instead of Lambda
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import cv2
import streamlit as st

# Custom L2Normalization layer to replace Lambda
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

# Function to find the image file recursively in subfolders
def find_image(article_id, base_directory):
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file == f"{article_id}.jpg":
                return os.path.join(root, file)  # Return the full path to the image
    return None  # Return None if the image is not found

# Create API connection
os.environ['KAGGLE_USERNAME'] = "reginaveliczky"
os.environ['KAGGLE_KEY'] = "4e6b26b7525700b49c336424d6ccc0b3"
api = KaggleApi()
api.authenticate()

# Download dataset
files = api.dataset_list_files('odins0n/handm-dataset-128x128').files
print("Available files in dataset:")
for f in files:
    print(f.name)

api.dataset_download_file('odins0n/handm-dataset-128x128', file_name='articles.csv', path=".")

# Datasets
df_a = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\df_a_images.csv")
df_c = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\customers.csv")
df_t = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\transactions_train.csv")

# Preprocess article descriptions
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Initialize tqdm for progress bars
tqdm.pandas()

def preprocess(x):
    x = ' ' + x + ' '
    x = x.lower()
    x = re.sub(r'\d+\-\d+', ' ', x)
    x = re.sub(r'\d+\.\d+', ' ', x)
    x = re.sub(r'\d+/\d+', ' ', x)
    x = re.sub(r'\d+', ' ', x)
    x = re.sub(r'\(.+?\)', ' ', x)
    x = re.sub(r' cm ', ' ', x)
    x = re.sub(r' denier ', ' ', x)
    x = re.sub(r' t\-shirt ', ' tshirt ', x)
    x = re.sub(r'(\w+)(\+)', r'\1 plus', x)
    x = re.sub(r'[^a-z\s]', ' ', x)
    tokens = x.strip().split()
    tokens = ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stopwords])
    return tokens

df_a['desc_full'] = df_a['product_type_name'] + ' ' + df_a['department_name'] + ' ' + df_a['colour_group_name'] + ' ' + df_a['section_name'] + ' ' + df_a['detail_desc'].fillna(' ')

# Use progress_apply instead of apply
df_a['desc_full_prep'] = df_a['desc_full'].progress_apply(preprocess)

# Embeddings
documents = [TaggedDocument(words=row['desc_full_prep'].split(), tags=[str(row['article_id'])]) for index, row in df_a[['article_id', 'desc_full_prep']].iterrows()]
doc2vec_model = Doc2Vec(documents, vector_size=200, window=5, min_count=2, workers=4, epochs=20)

doc2vec_dict = {}
for index, row in df_a[['article_id', 'desc_full_prep']].iterrows():
    doc2vec_dict[row['article_id']] = doc2vec_model.dv[str(row['article_id'])]

emb_features = np.array(list(doc2vec_dict.values()))

# One-hot encoding
ohe_columns = ['product_group_name', 'graphical_appearance_no', 'perceived_colour_master_id', 'index_code', 'garment_group_no']
ohe_features = pd.get_dummies(df_a[ohe_columns].astype(str)).astype(int).values
articles_features = np.concatenate([emb_features, ohe_features], axis=1)

# Mapping
articles_features_with_id = {}
for article_id, features in zip(df_a['article_id'].values, articles_features):
    articles_features_with_id[article_id] = features

# Customers
df_c['age'] = df_c['age'].fillna(df_c['age'].mean())

bins = [16, 20, 25, 30, 40, 50, 60, 100]
labels = [1, 2, 3, 4, 5, 6, 7]
df_c['age_bins'] = pd.cut(df_c['age'], bins=bins, labels=labels)

ohe_columns = ['club_member_status', 'fashion_news_frequency', 'age_bins']
ohe_features = pd.get_dummies(df_c[ohe_columns].astype(str)).astype(int)

# Debugging: print customer feature shapes
print(f"Customer features shape: {ohe_features.shape}")

customer_features = pd.concat([df_c[['FN', 'Active']], ohe_features], axis=1).values

# Fix the input shape of the model
input_user_shape = customer_features.shape[1]  # Dynamically adjust the input shape based on the number of features

customer_features_with_id = {}
for customer_id, features in zip(df_c['customer_id'].values, customer_features):
    customer_features_with_id[customer_id] = features

# Save features
np.save('articles_features.npy', articles_features_with_id)
np.save('customer_features.npy', customer_features_with_id)

# Handling Images
image_directory = "C:/Users/User/OneDrive/Desktop/PROJECT/images/"  # Directory with subfolders

# Use the find_image function to locate images within subfolders
df_a['image_name'] = df_a['article_id'].apply(lambda x: find_image(x, image_directory))

# Check if images were found and raise a warning for missing images
df_a['image_name'].apply(lambda x: st.warning(f"Image not found for article id: {x}") if x is None else None)

df_a_cleaned = df_a.dropna(subset=['image_name'])

# Plotting function
def plot_figures(figures, nrows=1, ncols=1, figsize=(8, 8)):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()

def load_image(image_path, resized_fac=0.1):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            w, h, _ = img.shape
            resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
            return resized
    return None

sampled_rows = df_a_cleaned.sample(6)
figures = {'im' + str(i): load_image(row['image_name']) for i, row in sampled_rows.iterrows() if load_image(row['image_name']) is not None}

if figures:
    plot_figures(figures, nrows=2, ncols=3)

# Train/test split
max_date = df_t['t_dat'].max()
end_train_date = datetime.strptime(max_date, '%Y-%m-%d') - timedelta(days=1)
transactions_train = df_t[df_t['t_dat'] <= end_train_date.strftime('%Y-%m-%d')]
transactions_test = df_t[df_t['t_dat'] > end_train_date.strftime('%Y-%m-%d')]

# Fix for SettingWithCopyWarning: Use .loc to modify DataFrame slice
transactions_test.loc[:, 'target'] = 1
transactions_train.loc[:, 'target'] = 1

# Negative sampling
articles_ids = pd.Series(list(articles_features_with_id.keys()))

def create_zero_target(transactions_df_grouped):
    customers_lst = []
    articles_lst = []
    for row in transactions_df_grouped[['customer_id', 'article_id']].itertuples():
        customer_id, article_ids = row[1], row[2]
        unord_article_id = articles_ids[~articles_ids.isin(article_ids)].sample(50)
        customers_lst.append(customer_id)
        articles_lst.append(unord_article_id)
    df = pd.DataFrame({'customer_id': customers_lst, 'article_id': articles_lst, 'target': [0] * len(customers_lst)}).explode('article_id')
    return df

# Test dataset creation
transactions_test_grouped = transactions_test.drop(['article_id'], axis=1).merge(df_t[['customer_id', 'article_id']], how='left', on='customer_id').groupby('customer_id')['article_id'].apply(lambda x: list(set(x))).reset_index()
zero_test_target = create_zero_target(transactions_test_grouped)
test = pd.concat([transactions_test, zero_test_target])[['customer_id', 'article_id', 'target']].sort_values(by='customer_id').reset_index(drop=True)

# Define shapes for customer and article features
default_customer_shape = customer_features_with_id[next(iter(customer_features_with_id))].shape
default_article_shape = articles_features_with_id[next(iter(articles_features_with_id))].shape

# Handle missing IDs
test_customer_features = np.array([customer_features_with_id.get(customer_id, np.zeros(default_customer_shape)) for customer_id in test['customer_id'].values])
test_article_features = np.array([articles_features_with_id.get(article_id, np.zeros(default_article_shape)) for article_id in test['article_id'].values])
test_target = test['target'].values

# Create train dataset
transactions_train_grouped = transactions_train.groupby('customer_id')['article_id'].apply(lambda x: len(set(x)))
selected_customers = transactions_train_grouped[transactions_train_grouped > 10].sample(20000, random_state=1).index.values
transactions_train = transactions_train[transactions_train['customer_id'].isin(selected_customers)]

train_customer_features = np.array([customer_features_with_id[customer_id] for customer_id in transactions_train['customer_id'].values])
train_article_features = np.array([articles_features_with_id[article_id] for article_id in transactions_train['article_id'].values])
train_target = transactions_train['target'].values

# Model setup
num_outputs = 32

user_NN = tf.keras.models.Sequential([    
    Dense(50, activation='relu'),
    Dense(40, activation='relu'),
    Dense(num_outputs) 
])

item_NN = tf.keras.models.Sequential([   
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_outputs)
])

# Use dynamic input shape based on actual customer features
input_user = Input(shape=(input_user_shape,))
vu = user_NN(input_user)
vu = L2Normalization()(vu)  # Use the custom L2Normalization layer instead of Lambda

input_item = Input(shape=(300,))
vm = item_NN(input_item)
vm = L2Normalization()(vm)  # Use the custom L2Normalization layer instead of Lambda

output = Dot(axes=1)([vu, vm])
model = tf.keras.Model(inputs=[input_user, input_item], outputs=output)

model.summary()

# Compile model
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

# Train model
model.fit([train_customer_features, train_article_features], train_target, epochs=10, validation_data=([test_customer_features, test_article_features], test_target))

# Save the trained model in Keras format
model.save('recommendation_model.keras')

print("Model saved as 'recommendation_model.keras'.")
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Dot, Layer  # Import Layer instead of Lambda
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import cv2
import streamlit as st

# Custom L2Normalization layer to replace Lambda
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

# Function to find the image file recursively in subfolders
def find_image(article_id, base_directory):
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file == f"{article_id}.jpg":
                return os.path.join(root, file)  # Return the full path to the image
    return None  # Return None if the image is not found

# Create API connection
os.environ['KAGGLE_USERNAME'] = "reginaveliczky"
os.environ['KAGGLE_KEY'] = "4e6b26b7525700b49c336424d6ccc0b3"
api = KaggleApi()
api.authenticate()

# Download dataset
files = api.dataset_list_files('odins0n/handm-dataset-128x128').files
print("Available files in dataset:")
for f in files:
    print(f.name)

api.dataset_download_file('odins0n/handm-dataset-128x128', file_name='articles.csv', path=".")

# Datasets
df_a = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\df_a_images.csv")
df_c = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\customers.csv")
df_t = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\transactions_train.csv")

# Preprocess article descriptions
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Initialize tqdm for progress bars
tqdm.pandas()

def preprocess(x):
    x = ' ' + x + ' '
    x = x.lower()
    x = re.sub(r'\d+\-\d+', ' ', x)
    x = re.sub(r'\d+\.\d+', ' ', x)
    x = re.sub(r'\d+/\d+', ' ', x)
    x = re.sub(r'\d+', ' ', x)
    x = re.sub(r'\(.+?\)', ' ', x)
    x = re.sub(r' cm ', ' ', x)
    x = re.sub(r' denier ', ' ', x)
    x = re.sub(r' t\-shirt ', ' tshirt ', x)
    x = re.sub(r'(\w+)(\+)', r'\1 plus', x)
    x = re.sub(r'[^a-z\s]', ' ', x)
    tokens = x.strip().split()
    tokens = ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stopwords])
    return tokens

df_a['desc_full'] = df_a['product_type_name'] + ' ' + df_a['department_name'] + ' ' + df_a['colour_group_name'] + ' ' + df_a['section_name'] + ' ' + df_a['detail_desc'].fillna(' ')

# Use progress_apply instead of apply
df_a['desc_full_prep'] = df_a['desc_full'].progress_apply(preprocess)

# Embeddings
documents = [TaggedDocument(words=row['desc_full_prep'].split(), tags=[str(row['article_id'])]) for index, row in df_a[['article_id', 'desc_full_prep']].iterrows()]
doc2vec_model = Doc2Vec(documents, vector_size=200, window=5, min_count=2, workers=4, epochs=20)

doc2vec_dict = {}
for index, row in df_a[['article_id', 'desc_full_prep']].iterrows():
    doc2vec_dict[row['article_id']] = doc2vec_model.dv[str(row['article_id'])]

emb_features = np.array(list(doc2vec_dict.values()))

# One-hot encoding
ohe_columns = ['product_group_name', 'graphical_appearance_no', 'perceived_colour_master_id', 'index_code', 'garment_group_no']
ohe_features = pd.get_dummies(df_a[ohe_columns].astype(str)).astype(int).values
articles_features = np.concatenate([emb_features, ohe_features], axis=1)

# Mapping
articles_features_with_id = {}
for article_id, features in zip(df_a['article_id'].values, articles_features):
    articles_features_with_id[article_id] = features

# Customers
df_c['age'] = df_c['age'].fillna(df_c['age'].mean())

bins = [16, 20, 25, 30, 40, 50, 60, 100]
labels = [1, 2, 3, 4, 5, 6, 7]
df_c['age_bins'] = pd.cut(df_c['age'], bins=bins, labels=labels)

ohe_columns = ['club_member_status', 'fashion_news_frequency', 'age_bins']
ohe_features = pd.get_dummies(df_c[ohe_columns].astype(str)).astype(int)

# Debugging: print customer feature shapes
print(f"Customer features shape: {ohe_features.shape}")

customer_features = pd.concat([df_c[['FN', 'Active']], ohe_features], axis=1).values

# Fix the input shape of the model
input_user_shape = customer_features.shape[1]  # Dynamically adjust the input shape based on the number of features

customer_features_with_id = {}
for customer_id, features in zip(df_c['customer_id'].values, customer_features):
    customer_features_with_id[customer_id] = features

# Save features
np.save('articles_features.npy', articles_features_with_id)
np.save('customer_features.npy', customer_features_with_id)

# Handling Images
image_directory = "C:/Users/User/OneDrive/Desktop/PROJECT/images/"  

# Use the find_image function to locate images within subfolders
df_a['image_name'] = df_a['article_id'].apply(lambda x: find_image(x, image_directory))

# Check if images were found and raise a warning for missing images
df_a['image_name'].apply(lambda x: st.warning(f"Image not found for article id: {x}") if x is None else None)

df_a_cleaned = df_a.dropna(subset=['image_name'])

# Plotting function
def plot_figures(figures, nrows=1, ncols=1, figsize=(8, 8)):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()

def load_image(image_path, resized_fac=0.1):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            w, h, _ = img.shape
            resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
            return resized
    return None

sampled_rows = df_a_cleaned.sample(6)
figures = {'im' + str(i): load_image(row['image_name']) for i, row in sampled_rows.iterrows() if load_image(row['image_name']) is not None}

if figures:
    plot_figures(figures, nrows=2, ncols=3)

# Train/test split
max_date = df_t['t_dat'].max()
end_train_date = datetime.strptime(max_date, '%Y-%m-%d') - timedelta(days=1)
transactions_train = df_t[df_t['t_dat'] <= end_train_date.strftime('%Y-%m-%d')]
transactions_test = df_t[df_t['t_dat'] > end_train_date.strftime('%Y-%m-%d')]

# Fix for SettingWithCopyWarning: Use .loc to modify DataFrame slice
transactions_test.loc[:, 'target'] = 1
transactions_train.loc[:, 'target'] = 1

# Negative sampling
articles_ids = pd.Series(list(articles_features_with_id.keys()))

def create_zero_target(transactions_df_grouped):
    customers_lst = []
    articles_lst = []
    for row in transactions_df_grouped[['customer_id', 'article_id']].itertuples():
        customer_id, article_ids = row[1], row[2]
        unord_article_id = articles_ids[~articles_ids.isin(article_ids)].sample(50)
        customers_lst.append(customer_id)
        articles_lst.append(unord_article_id)
    df = pd.DataFrame({'customer_id': customers_lst, 'article_id': articles_lst, 'target': [0] * len(customers_lst)}).explode('article_id')
    return df

# Test dataset creation
transactions_test_grouped = transactions_test.drop(['article_id'], axis=1).merge(df_t[['customer_id', 'article_id']], how='left', on='customer_id').groupby('customer_id')['article_id'].apply(lambda x: list(set(x))).reset_index()
zero_test_target = create_zero_target(transactions_test_grouped)
test = pd.concat([transactions_test, zero_test_target])[['customer_id', 'article_id', 'target']].sort_values(by='customer_id').reset_index(drop=True)

# Define shapes for customer and article features
default_customer_shape = customer_features_with_id[next(iter(customer_features_with_id))].shape
default_article_shape = articles_features_with_id[next(iter(articles_features_with_id))].shape

# Handle missing IDs
test_customer_features = np.array([customer_features_with_id.get(customer_id, np.zeros(default_customer_shape)) for customer_id in test['customer_id'].values])
test_article_features = np.array([articles_features_with_id.get(article_id, np.zeros(default_article_shape)) for article_id in test['article_id'].values])
test_target = test['target'].values

# Create train dataset
transactions_train_grouped = transactions_train.groupby('customer_id')['article_id'].apply(lambda x: len(set(x)))
selected_customers = transactions_train_grouped[transactions_train_grouped > 10].sample(20000, random_state=1).index.values
transactions_train = transactions_train[transactions_train['customer_id'].isin(selected_customers)]

train_customer_features = np.array([customer_features_with_id[customer_id] for customer_id in transactions_train['customer_id'].values])
train_article_features = np.array([articles_features_with_id[article_id] for article_id in transactions_train['article_id'].values])
train_target = transactions_train['target'].values

# Model setup
num_outputs = 32

user_NN = tf.keras.models.Sequential([    
    Dense(50, activation='relu'),
    Dense(40, activation='relu'),
    Dense(num_outputs) 
])

item_NN = tf.keras.models.Sequential([   
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_outputs)
])

# Use dynamic input shape based on actual customer features
input_user = Input(shape=(input_user_shape,))
vu = user_NN(input_user)
vu = L2Normalization()(vu)  # Use the custom L2Normalization layer instead of Lambda

input_item = Input(shape=(300,))
vm = item_NN(input_item)
vm = L2Normalization()(vm)  # Use the custom L2Normalization layer instead of Lambda

output = Dot(axes=1)([vu, vm])
model = tf.keras.Model(inputs=[input_user, input_item], outputs=output)

model.summary()

# Compile model
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

# Train model
model.fit([train_customer_features, train_article_features], train_target, epochs=10, validation_data=([test_customer_features, test_article_features], test_target))

# Save the trained model in Keras format
model.save('recommendation_model.keras')

print("Model saved as 'recommendation_model.keras'.")
