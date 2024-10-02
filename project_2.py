#!/usr/bin/env python
# coding: utf-8

# In[219]:


import os
import pandas as pd
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dot, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import time
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import cv2


# # Datasets

# In[154]:


df_a=pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\df_a_images.csv")


# In[5]:


df_c=pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\customers.csv")


# In[6]:


df_t = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\PROJECT\transactions_train.csv")


# # Articles

# In[155]:


df_a.shape


# In[156]:


df_a.isnull().sum()/len(df_a)


# In[157]:


df_a.nunique()


# In[158]:


df_a[['product_type_no','product_type_name']].sample(5)


# In[159]:


df_a.product_group_name.unique()


# In[160]:


# one hot endcoding
ohe_columns = ['product_group_name', 'graphical_appearance_no','perceived_colour_master_id', 'index_code', 
               'garment_group_no']
# embeddings
emb_columns = ['product_type_name', 'department_name', 'colour_group_name', 'section_name', 'detail_desc']


# In[161]:


df_a[['article_id']+emb_columns].head()


# In[162]:


# categorical data "cleaning"

nltk.download('stopwords')

stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[163]:


nltk.download('wordnet')

def preprocess(x):
    x = ' '+x+' '
    x = x.lower()
    
    # remove numbers
    x = re.sub(r'\d+\-\d+', ' ', x)
    x = re.sub(r'\d+\.\d+', ' ', x)
    x = re.sub(r'\d+/\d+', ' ', x)
    x = re.sub(r'\d+', ' ', x)
    
    # remove everything in braces
    x = re.sub(r'\(.+?\)', ' ', x)   # .=any character, +=one or more, ?=as least as possible, stopping at the first closing p.
    
    x = re.sub(r' cm ', ' ', x)
    x = re.sub(r' denier ', ' ', x)
    x = re.sub(r' t\-shirt ', ' tshirt ', x)
    x = re.sub(r'(\w+)(\+)', r'\1 plus', x)
    
    # remove everything except letters
    x = re.sub(r'[^a-z\s]', ' ', x)
    tokens = x.strip().split()
    tokens = ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stopwords])
    return tokens

s = 'Metal hair clips in various designs, two of which are slides. Length from 6.5 cm to 7 cm .'
preprocess(s)


# In[164]:


tqdm.pandas()
df_a['desc_full'] = df_a['product_type_name']+' '+df_a['department_name']+' '\
                        +df_a['colour_group_name']+' '+df_a['section_name']+' '\
                        +df_a['detail_desc'].fillna(' ')
df_a['desc_full_prep'] = df_a['desc_full'].progress_apply(preprocess)


# In[165]:


texts = df_a['desc_full_prep'].unique().tolist()
with open('texts_normalized.txt', 'w') as f:
    for text in texts:
        f.write(text)
        f.write('\n')


# In[166]:


#unsupervised learning, a Model that represents each Document as a Vector
#EMB_features creation

# Prepare tagged documents
documents = [TaggedDocument(words=row['desc_full_prep'].split(), tags=[str(row['article_id'])]) 
             for index, row in df_a[['article_id', 'desc_full_prep']].iterrows()]

# Train the Doc2Vec model, these are the recommended settings
doc2vec_model = Doc2Vec(documents, vector_size=200, window=5, min_count=2, workers=4, epochs=20)


# Create dictionary of document vectors
doc2vec_dict = {}
for index, row in df_a[['article_id', 'desc_full_prep']].iterrows():
    doc2vec_dict[row['article_id']] = doc2vec_model.dv[str(row['article_id'])]  # Fetching the vector for the document

# Convert to numpy array and check shape
emb_features = np.array(list(doc2vec_dict.values()))
print(emb_features.shape)


# In[167]:


ohe_features = pd.get_dummies(df_a[ohe_columns].astype(str)).astype(int).values
ohe_features.shape


# In[168]:


articles_features = np.concatenate([emb_features, ohe_features], axis=1)
articles_features.shape


# In[169]:


# mapping the article_id to its corresponding feature vector (the value)

articles_features_with_id = {}
for article_id, features in zip(df_a['article_id'].values, articles_features):
    articles_features_with_id[article_id] = features


# In[170]:


# Check for missing values in the last column
missing_in_last_column = df_a.isna().sum().iloc[-1]
print(f"Total missing values in the last column: {missing_in_last_column}")

# Replace missing values
df_a.fillna(value="No Description", inplace=True)

# Define the function to adjust the article ID to 10 characters
def adjust_id(x):
    return str(x).zfill(10)  

# Adjust the article ID and create product code
df_a["article_id"] = df_a["article_id"].apply(lambda x: adjust_id(x))
df_a["product_code"] = df_a["article_id"].apply(lambda x: x[:3])


# # Customers

# In[171]:


#FN-fashion news

df_c.shape


# In[172]:


df_c.isnull().sum()/len(df_c)


# In[173]:


df_c.nunique()


# In[174]:


df_c['age'].describe()


# In[175]:


df_c[['FN', 'Active']] = df_c[['FN', 'Active']].fillna(0).astype(int)
df_c[['club_member_status', 'fashion_news_frequency']] = df_c[['club_member_status','fashion_news_frequency']].fillna('NONE')


# In[176]:


df_c['age'] = df_c['age'].fillna(df_c['age'].mean())

bins = [16, 20, 25, 30, 40, 50, 60, 100]
labels = [1,2,3,4,5,6,7]
df_c['age_bins'] = pd.cut(df_c['age'], bins=bins, labels=labels)
df_c.head()


# In[177]:


ohe_columns = ['club_member_status', 'fashion_news_frequency', 'age_bins']
ohe_features = pd.get_dummies(df_c[ohe_columns].astype(str)).astype(int)
customer_features = pd.concat([df_c[['FN', 'Active']], ohe_features], axis=1).values
customer_features.shape


# In[178]:


customer_features_with_id = {}
for customer_id, features in zip(df_c['customer_id'].values, customer_features):
    customer_features_with_id[customer_id] = features


# # Transaction

# In[224]:


df_t.shape


# In[225]:


df_t.isnull().sum()/len(df_t)


# In[226]:


df_t.nunique()


# In[227]:


# Sort the dates in ascending order and plot the data
pd.to_datetime(df_t['t_dat']).value_counts().sort_index().plot(kind='line', figsize=(20, 7))


# In[228]:


# Printing Maximum, Minimum, and Average Prices
print("Maximum Price is:", df_t["price"].max(), "\n" +
      "Minimum Price is:", df_t["price"].min(), "\n" +
      "Average Price is:", df_t["price"].mean())


# In[229]:


# Grouping the data
basket = df_t.groupby("customer_id").agg({'article_id':'count', 
                                          'price': 'sum'}).reset_index()
basket.columns = ["customer_id", "units", "order_price"]

# Replace inf values with NaN 
basket.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
basket.dropna(inplace=True)

# Printing Order Statistics
print("UNITS/ORDER")
print("Maximum Units/Order is:", basket["units"].max(), "\n" +
      "Minimum Units/Order is:", basket["units"].min(), "\n" +
      "Average Units/Order is:", basket["units"].mean(), "\n")

print("SPENDING/ORDER")
print("Maximum Spending/Order is:", basket["order_price"].max(), "\n" +
      "Minimum Spending/Order is:", basket["order_price"].min(), "\n" +
      "Average Spending/Order is:", basket["order_price"].mean())


# # Images

# In[232]:


image_folder= "C:/Users/User/OneDrive/Desktop/PROJECT/images"
image_extension = ".jpg"


# In[233]:


"""# function to search through subfolders for the image
def get_image_path(article_id):
    image_name = f"{article_id}{image_extension}"
    for root, dirs, files in os.walk(image_folder):
        if image_name in files:
            return os.path.join(root, image_name)
    return None  # Return None if the image isn't found in any subfolder

# Apply the function to the article_id column and create a new column with the image path
df_a['image_path'] = df_a['article_id'].apply(get_image_path)

df_a.head()
"""
image_extension=".jpg"

df_a['image_name'] = df_a['article_id'].apply(lambda x: f"{x}{image_extension}")


# In[235]:


# Count null values in the 'image_path' column
null_count = df_a["image_path"].isnull().sum()
print(f"Number of null values in 'image_path': {null_count}")

# Remove rows where 'image_path' is null
df_a_cleaned = df_a.dropna(subset=['image_path'])

# Check the number of rows after cleaning
print(f"Number of rows after cleaning: {len(df_a_cleaned)}")


# In[236]:


import os
import pandas as pd

# First, clean the DataFrame to remove any rows where 'image_path' is NaN or invalid
df_a_cleaned = df_a.dropna(subset=['image_path'])  # Drop rows where 'image_path' is NaN
df_a_cleaned = df_a_cleaned[df_a_cleaned['image_path'].str.strip() != '']  # Remove empty strings
df_a_cleaned = df_a_cleaned[df_a_cleaned['image_path'].str.lower() != 'nan']  # Remove 'nan' strings

def plot_figures(figures, nrows=1, ncols=1, figsize=(8, 8)):
    """Plot a dictionary of figures."""
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()

def load_image(image_path, resized_fac=0.1):
    """Load and resize an image if it exists."""
    if os.path.exists(image_path):  # Check if the file actually exists
        img = cv2.imread(image_path)
        if img is not None:
            w, h, _ = img.shape
            resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
            return resized
        else:
            print(f"Image could not be loaded: {image_path}")
    else:
        print(f"File does not exist: {image_path}")
    return None

# Sample rows from the cleaned DataFrame
sampled_rows = df_a_cleaned.sample(6)

# Generate the dictionary of titles and loaded images, only including valid images
figures = {'im' + str(i): load_image(row['image_path']) for i, row in sampled_rows.iterrows() if load_image(row['image_path']) is not None}

# Ensure that only rows with valid images are included
if figures:
    # Plot the images with 2 rows and 3 columns (adjust based on sample size)
    plot_figures(figures, nrows=2, ncols=3)
else:
    print("No valid images found to display.")


# # Train test split

# In[237]:


max_date = df_t['t_dat'].max()

# cut off,keeping one last 1 day for test dataset
end_train_date = datetime.strptime(max_date, '%Y-%M-%d') - timedelta(days=1)
end_train_date = end_train_date.strftime('%Y-%M-%d')
end_train_date


# In[238]:


transactions_train = df_t[df_t['t_dat']<=end_train_date]
transactions_test = df_t[df_t['t_dat']>end_train_date]
transactions_train.shape, transactions_test.shape


# In[239]:


#This function generates negative samples for each customer, it assigns articles to a customer that they did not interact with

articles_ids = pd.Series(list(articles_features_with_id.keys()))

def create_zero_target(transactions_df_grouped):
    cutomers_lst = []
    articles_lst = []
    for row in transactions_df_grouped[['customer_id', 'article_id']].itertuples():
        customer_id, article_ids = row[1], row[2]
        unord_article_id = articles_ids[~articles_ids.isin(article_ids)].sample(50) #filtering the data the cust interacted with
        cutomers_lst.append(customer_id)
        articles_lst.append(unord_article_id) #adds the sampled negative 50 to the articles_lst
    df = pd.DataFrame({'customer_id':cutomers_lst, 'article_id':articles_lst, 'target':[0]*len(cutomers_lst)}).\
                                                                                            explode('article_id')
    #each row now represents a single customer-article pair - explode
    return df


# # Create test dataset

# In[240]:


transactions_test.loc[:, 'target'] = 1


# In[241]:


#creating the final test database with positive and negative samples

transactions_test_grouped = transactions_test.drop(['article_id'], axis=1).\
                                                    merge(df_t[['customer_id', 'article_id']], \
                                                    how='left', on='customer_id').\
                                                    groupby('customer_id')['article_id'].apply(lambda x: \
                                                                                        list(set(x))).reset_index()
zero_test_target = create_zero_target(transactions_test_grouped)
test = pd.concat([transactions_test, zero_test_target])[['customer_id', 'article_id', 'target']].\
                                                                    sort_values(by='customer_id').reset_index(drop=True)
test


# In[242]:


# Define default shapes for customer and article features if keys are missing
default_customer_shape = customer_features_with_id[next(iter(customer_features_with_id))].shape # iter allows you to loop through the keys of the dictionary
default_article_shape = articles_features_with_id[next(iter(articles_features_with_id))].shape #next() retrieves the first key from the iterator

# Handle missing customer IDs 
test_customer_features = np.array([
    customer_features_with_id.get(customer_id, np.zeros(default_customer_shape)) 
    for customer_id in test['customer_id'].values
])

# Handle missing article IDs 
test_article_features = np.array([
    articles_features_with_id.get(article_id, np.zeros(default_article_shape)) 
    for article_id in test['article_id'].values
])

# Extract target values
test_target = test['target'].values

# Print the shapes of the arrays
test_customer_features.shape, test_article_features.shape, test_target.shape


# # Create train dataset

# In[244]:


#counts the number of unique article_ids that the customer has interacted with
transactions_train_grouped = transactions_train.groupby('customer_id')['article_id'].apply(lambda x: len(set(x)))
#customers who have interacted with more than 10 unique articles, random 20000 sample
selected_customers = transactions_train_grouped[transactions_train_grouped>10].sample(20000, random_state=1).index.values
#keeping only these customers for the data frame
transactions_train = transactions_train[transactions_train['customer_id'].isin(selected_customers)]
transactions_train.shape


# In[245]:


transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])
#most recent date
dt_train_max = transactions_train['t_dat'].max()
#rating dictionary based on transaction recency
rating_dict = {x: 0.5**((dt_train_max - x).days/360) for x in transactions_train['t_dat'].unique()}
#mapping the values in the dataframe
transactions_train['target'] = transactions_train['t_dat'].map(rating_dict)
transactions_train.head()


# In[246]:


# Drop the 'article_id' column and merge with df_t on 'customer_id'
transactions_train_dropped = transactions_train.drop(['article_id'], axis=1)
transactions_train_merged = transactions_train_dropped.merge(
    df_t[['customer_id', 'article_id']], how='left', on='customer_id'
)

# Group by 'customer_id' and collect unique article_ids
transactions_train_grouped = transactions_train_merged.groupby('customer_id')['article_id'].agg(set).reset_index()

# Convert the set back to a list with a progress bar
transactions_train_grouped['article_id'] = transactions_train_grouped['article_id'].progress_apply(list)

# Create zero target based on the grouped data
zero_train_target = create_zero_target(transactions_train_grouped)

# Concatenate and sort the data
train = pd.concat([transactions_train, zero_train_target])[['customer_id', 'article_id', 'target']]
train = train.sort_values(by='customer_id').reset_index(drop=True)

train


# In[247]:


#creating array that contains features for each customer in the training set
train_customer_features = np.array([customer_features_with_id[customer_id] for customer_id in \
                                    train['customer_id'].values])
##creating array that contains features for each article in the training set
train_article_features = np.array([articles_features_with_id[article_id] for article_id in \
                                   train['article_id'].values])
#extracting the target values from the train dataset
train_target = train['target'].values
train_customer_features.shape, train_article_features.shape, train_target.shape


# In[249]:


#memory cleanup
del transactions_train
del transactions_train_grouped
del transactions_test
del transactions_test_grouped
import gc
gc.collect()


# # Train

# In[250]:


num_outputs = 32

# User neural network
user_NN = tf.keras.models.Sequential([    
    Dense(50, activation='relu'), #Rectified Linear Unit  - activation system that outputs the input directly if it is positive, otherwise, it outputs zero
    Dense(40, activation='relu'),
    Dense(num_outputs) 
])

# Item neural network
item_NN = tf.keras.models.Sequential([   
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_outputs)
])

# Create the user input and point to the base network
input_user = Input(shape=(17,))  # 17 customer features
vu = user_NN(input_user) #Passes this input through the user neural network to get a vector

# Apply L2 normalization using Lambda layer, L2 normalization scales the vector such that its Euclidean length is 1
vu = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)

# Create the item input and point to the base network
input_item = Input(shape=(300,))  # 300 article features
vm = item_NN(input_item) #Passes input through the item neural network to get a vector

# Apply L2 normalization using Lambda layer
vm = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)

# Compute the dot product of the two vectors vu and vm ( Interaction between User and Item)
output = Dot(axes=1)([vu, vm])

# Model Definition: Specify the inputs and output of the model
model = tf.keras.Model(inputs=[input_user, input_item], outputs=output)

# Model summary
model.summary()


# In[251]:


print(train_customer_features.shape)
print(train_article_features.shape)


# In[252]:


#specifying which metric to track and whether you are looking for the maximum or minimum value of that metric.
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')
            
#checking if the current epoch has produced a better value for the monitored metric
    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric] #retrieves the current value of the specified metric from the logs dictionary
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()
                
save_best_model = SaveBestModel()


# In[253]:


#By setting a seed, you ensure that the randomness in operations is deterministic and reproducible across runs
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError() #loss function
#defining the optimizer, which controls how the model's weights are updated during training
opt = keras.optimizers.Adam(learning_rate=0.01)
#preparing the model for training, associating the optimizer and loss function with it
model.compile(optimizer=opt,
              loss=cost_fn)


# In[254]:


tf.random.set_seed(1)

model.fit([train_customer_features, train_article_features], train_target, epochs=2, \
          validation_data=([test_customer_features, test_article_features], test_target), \
          callbacks=[save_best_model])
#set best weigts
model.set_weights(save_best_model.best_weights)


# In[209]:


"""np.save('model.npy', np.empty(5))

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
"""
model = Sequential([
    # Adjust input shape to (317,) to match your concatenated input
    Dense(32, input_shape=(317,), activation='relu'),
    
    # Apply L2 normalization with the updated input shape
    Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1), output_shape=(32,)),
    
    # Output layer for binary classification
    Dense(1, activation='sigmoid')
])


# In[230]:


"""model = pickle.load(open("C:/Users/User/OneDrive/Desktop/PROJECT/model.pkl", 'rb'))"""


# In[212]:


del train_customer_features
del train_article_features
del train_target
import gc
gc.collect()


# # Predict on single example

# In[255]:


#creating a subset of the test dataset by filtering it to include only the rows where the customer_id matches a specific value
test_subset = test[test['customer_id']==test['customer_id'].iloc[1]]
test_subset.head()


# In[256]:


#creating two arrays
test_subset_customer_features = np.array([customer_features_with_id[customer_id] for customer_id in \
                                   test_subset['customer_id'].values])
test_subset_article_features = np.array([articles_features_with_id[article_id] for article_id in \
                                  test_subset['article_id'].values])


# In[261]:


#make predictions on the test subset
#y_p=predicted output from the model
y_p = model.predict([test_subset_customer_features, test_subset_article_features])
y_p[:5]


# In[262]:


#sorting the predictions
sorted_index = np.argsort(-y_p,axis=0).reshape(-1).tolist()  
#reordering the True Target Values
sorted_y = test_subset['target'].values[sorted_index]
sorted_y, np.array(sorted_index)


# # Inference

# In[263]:


#generating predicted scores for customer-article pairs
predicted_score = model.predict([test_customer_features, test_article_features])


# In[264]:


test['predicted_score'] = predicted_score.reshape(-1)
#Grouping Data by Customer
test_grouped = test.groupby('customer_id').agg({'article_id':list, 'target':list, 'predicted_score':list})
#Filtering Customers with Varying Targets
test_grouped = test_grouped[test_grouped['target'].apply(lambda x: len(set(x))>1)]
#Sorting Articles by Predicted Score
test_grouped['predicted_article_id'] = test_grouped[['article_id', 'predicted_score']].\
                                    apply(lambda x: list(np.array(x[0])[np.argsort(x[1])[::-1]]), axis=1)
test_grouped = test_grouped.reset_index()


# In[267]:


#keeping only the articles with a predicted score of 0.7 or higher
def filter_high_predictions(row):
    article_ids = np.array(row['article_id'])
    predicted_scores = np.array(row['predicted_score'])
    mask = predicted_scores >= 0.5
    filtered_article_ids = article_ids[mask]
    filtered_predicted_scores = predicted_scores[mask]
    return filtered_article_ids, filtered_predicted_scores

test_grouped['filtered_article_id'], test_grouped['filtered_predicted_score'] = \
    zip(*test_grouped.apply(filter_high_predictions, axis=1))

test_grouped = test_grouped[test_grouped['filtered_article_id'].apply(len) > 0]


# In[268]:


# Display filtered results
test_grouped[['customer_id', 'filtered_article_id', 'filtered_predicted_score']]


# # Evaluate results

# In[269]:


#evaluating how many relevant items are in the top-k recommendations
def ap_k(actual, predicted, k):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)
#ranking metric that accounts for both the relevance of items and their position in the ranking metrics
def ndcg_k(y_true, y_pred, k):
    dcg = 0
    N = len(y_true)
    if N == 0:
        return 0
    for idx, item in enumerate(y_pred[:k], start=1):
        dcg += (2 ** (item in y_true) - 1) / np.log2(1 + idx)
    idcg = sum(1 / np.log2(1 + i) for i in range(1, N + 1))
    return dcg / idcg

def measure_rank_metrics(test_set, k):
    
    group_cols = ['customer_id']
    explode_cols = ['article_id', 'target']
    data_full = test_set[group_cols + explode_cols +['predicted_article_id']].explode(explode_cols)
    data = data_full.sort_values(group_cols, ascending=False)
    # для сокращения числа строк
    data[f'rn'] = data.groupby(group_cols).cumcount()+1
    data = data[(data.rn <= k) | (data.target == 1)]

    data_group = data.groupby(group_cols)
    map_all = round(data_group.apply(
                lambda x: ap_k(x.loc[x['target'] == 1, 'article_id'].tolist(), 
                              x['predicted_article_id'].values[0][:k], k)).values.mean(), 5)

    ndcg_all = round(data_group.apply(
                lambda x: ndcg_k(x.loc[x['target'] == 1, 'article_id'].tolist(), 
                               x['predicted_article_id'].values[0][:k], k)).values.mean(), 5)
    
    return map_all, ndcg_all

def measure_metrics(test_set, k=10):
    metrics = {}
    map_all, ndcg_all = measure_rank_metrics(test_set, k)
    metrics['map_all'] = map_all
    metrics['ndcg_all'] = ndcg_all
    
    return metrics


# In[270]:


measure_metrics(test_grouped, 12)


# In[271]:


measure_metrics(test_grouped, 20)


# In[272]:


measure_metrics(test_grouped, 200)   #average precision & normalized


# In[ ]:


del test_customer_features
del test_article_features
del test_target
import gc
gc.collect()


# # Suggest popular items for new customers 

# In[ ]:


# get last 30 days
start_popular_date = datetime.strptime(max_date, '%Y-%m-%d') - timedelta(days=30)
start_popular_date = start_popular_date.strftime('%Y-%m-%d')
start_popular_date


# In[ ]:


# find transactions from the last 30 days and then extracts the most popular articles
transactions_popular = df_t[df_t['t_dat']>=start_popular_date]
popular_goods = transactions_popular['article_id'].value_counts().head(12).index.tolist()
popular_goods = ' '.join([str(item) for item in popular_goods])
popular_goods

