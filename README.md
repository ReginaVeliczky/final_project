# final_project

# Fashion Recommendation System with Streamlit Interface

This project demonstrates a recommendation system that suggests personalized fashion items for users based on customer and article data. It leverages a neural network model integrated with a Streamlit interface to allow users to interact and view recommended products. The model is trained on data from the **H&M Dataset** and utilizes article descriptions, customer features, and transaction history for its recommendations.

## About the Project

This recommendation system is designed to predict fashion items that are most likely to interest a user, based on their previous shopping habits and customer information. The model processes the article data, including textual product descriptions, and learns the preferences of users using transaction records.

Key features include:
- **Personalized Recommendations**: Each user gets custom recommendations tailored to their tastes.
- **Article Embedding**: Product descriptions are transformed into vectors using a Doc2Vec model for efficient matching.
- **Streamlit Interface**: Users can interact with the model in real-time via a clean and simple web interface.
- **Image Display**: Product images are shown alongside recommendations for a more engaging experience.

### Built With

- [TensorFlow] - Deep Learning Framework
- [Streamlit] - Web Framework
- [Pandas]- Data Manipulation
- [NumPy]- Numerical Computing
- [Kaggle API]- Dataset API
- [OpenCV] - Image Processing

## Getting Started

To get a local copy up and running, follow these simple steps:

### Prerequisites

Ensure you have the following:
- Python 3.x installed
- Kaggle API account and credentials setup

### Installation

1. Clone the repo:

```bash
git clone https://github.com/your-username/fashion-recommendation-system.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Download the dataset:

Set up your Kaggle API credentials by adding your `KAGGLE_USERNAME` and `KAGGLE_KEY` to your environment variables, then run:

```bash
python download_data.py
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

Once the Streamlit application is running, you will be presented with a simple interface. The user can enter their **H&M membership number**, and the system will generate personalized recommendations based on the data.

- Enter your membership number to get the recommendations.
- Recommended articles will be displayed with their corresponding images.
- Users can provide feedback on their shopping experience through a slider input.

## Model Overview

The recommendation model is built using TensorFlow. It consists of two primary neural networks:
- **Customer Network**: Learns customer preferences based on features such as age, membership status, and shopping habits.
- **Article Network**: Embeds articles using both textual descriptions and one-hot encoded categorical features.

These networks map both customers and articles into a shared vector space, where recommendations are made by comparing the similarity between customer and article embeddings.

The model was trained on transaction data and optimized using the Mean Squared Error (MSE) loss function with an Adam optimizer.

## Streamlit Features

The Streamlit interface offers the following functionality:
- **Customer Input**: Users can input their membership number to fetch recommendations.
- **Recommendation Display**: The system suggests top articles with corresponding images.
- **User Feedback**: Customers can rate their shopping experience via sliders.
- **Membership Sign-up**: Non-members can follow the link to sign up on the H&M website.

## Future Enhancements

- **Collaborative Filtering**: Adding collaborative filtering to consider user-item interactions.
- **Improved UI**: Enhance the user interface for better interaction 
- **Real-Time Feedback**: Incorporate user feedback to further personalize recommendations.
- **Advanced Search**: Allow users to filter recommendations by category, price range, etc.

