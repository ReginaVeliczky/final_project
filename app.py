import streamlit as st

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

# URL of the logo image
logo_url = "https://upload.wikimedia.org/wikipedia/commons/5/53/H%26M-Logo.svg"

# Add the logo and title, center both
st.markdown(f"""
    <div style="text-align: center;">
        <img src="{logo_url}" alt="H&M Logo" style="width: 150px; height: auto; margin-right: 20px;">
        <h1 class="title" style="display: inline-block; font-size: 45px; vertical-align: middle;">Christmas Giveaway</h1>
    </div>
    """, unsafe_allow_html=True)



# URL of the image
image_url = "https://parc-le-breos.co.uk/wp-content/uploads/2017/09/Christmas-Bow-PNG-Image.png"

# Display the image below the title, adjusted to the width of the screen
st.image(image_url, caption=None, use_column_width=True)

# Header
st.markdown('<h2 class="header">"Experience more this Christmas"</h2>', unsafe_allow_html=True)

st.markdown("""
    <style>
    .space {
        margin-top: 50px;
    }
    </style>
    <div class="space"></div>
    """, unsafe_allow_html=True)

# URL of the image
image_url2 = "https://e7852c3a.rocketcdn.me/wp-content/uploads/2015/11/picmonkey-collage10.jpg"

# Display the image in the app, adjusted to the width of the screen
st.image(image_url2, caption=None, use_column_width=True)

st.markdown("""
    <style>
    .space {
        margin-top: 50px;
    }
    </style>
    <div class="space"></div>
    """, unsafe_allow_html=True)

# Subheader 2 (Are you an H&M member?)
st.markdown('<h2 class="subheader-member">Are you an H&M member?</h2>', unsafe_allow_html=True)

# Predefined correct membership number for validation
correct_membership_number = "123456"  # Replace with the actual correct number

# Center buttons with custom styling
st.markdown("""
    <style>
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
    }

    .btn-black, .btn-red {
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-family: 'Montserrat', sans-serif;
        font-size: 16px;
        cursor: pointer;
    }

    .btn-black {
        background-color: black;
        color: white;
    }

    .btn-red {
        background-color: #f63366;
        color: white;
    }

    .btn-black:hover {
        background-color: #333;
    }

    .btn-red:hover {
        background-color: #cc2a54;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a form for the first button and the input field
with st.form(key='member_form'):
    # Form elements
    membership_number = st.text_input("If yes,please type your H&M membership number below:")
    
    # Form submission button
    submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        # Validate membership number when the user presses the submit button
        if membership_number == correct_membership_number:
            st.success("You are successfully signed in, please answer the following questions.")
        else:
            st.error("Unfortunately, the number is incorrect, please try again!")

# Second button for non-membership link
st.markdown("""
    <div class="button-container">
        <a href="https://www2.hm.com/en_gb/member/info.html" target="_blank">
            <button class="btn-red">Not yet, I want to join!</button>
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .space {
        margin-top: 50px;
    }
    </style>
    <div class="space"></div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("""
    <style>
    .subheader-small {
        font-family: 'Montserrat', sans-serif;
        font-size: 18px;  
        color: black;
        text-align: center;
    }
    </style>
    <h2 class="subheader-small">Please answer the following questions to enter the competition:</h2>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .space {
        margin-top: 50px;
    }
    </style>
    <div class="space"></div>
    """, unsafe_allow_html=True)

# First question
st.markdown(
    """
    <style>
    .custom-container {
        width: 60%;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Wrap your slider within the custom container
with st.container():
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.slider("How satisfied are you with the overall shopping experience at H&M?", 1, 10, 10)
    st.markdown('</div>', unsafe_allow_html=True)

# Second question
container = st.container()
container.slider("How satisfied are you with the quality and variety of H&M products?", 1, 10, 10)

# Third question
container = st.container()
container.slider("How would you rate the customer service you receive at H&M?", 1, 10, 10)

st.checkbox("By clicking here, I state that I have read and understood the terms and conditions*.")

# Submit button
if st.button("Submit"):

    # Add some spacing after the button is clicked
    st.markdown("""
        <style>
        .space {
            margin-top: 50px;
        }
        </style>
        <div class="space"></div>
        """, unsafe_allow_html=True)

    # Display the message in Montserrat font
    st.markdown("""
        <div style="text-align: center; font-size: 20px; font-family: 'Montserrat', sans-serif;">
            <strong>We are grateful to our members for spending another year with us.</strong><br>
            <strong>As a token of our appreciation, we would like to gift a few of you with an item of your choice.</strong><br>
        </div>
        """, unsafe_allow_html=True)

    # URL of the image
    image_url = "https://imageio.forbes.com/specials-images/imageserve/5fb5c024bf3bf4129d04e891/Red-and-Green-Gift/960x0.jpg?format=jpg&width=1440"

    # Center the image and adjust the size
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="{image_url}" alt="Red and Green Gift" style="width: 50%; height: auto;">
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

    # More content in Montserrat font
    st.markdown("""
        <div style="text-align: center; font-size: 20px; font-family: 'Montserrat', sans-serif;">
            <strong>Below are the items we have personally selected for you.</strong><br>
            <strong>Choose your favorite, and if you're lucky, it will be under your Christmas tree.</strong><br>
        </div>
        """, unsafe_allow_html=True)

    # Add some more spacing
    st.markdown("""
        <style>
        .space {
            margin-top: 50px;
        }
        </style>
        <div class="space"></div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Add checkboxes to each column
    with col1:
        checkbox1 = st.checkbox("Select this item", key="checkbox1")
    with col2:
        checkbox2 = st.checkbox("Select this item", key="checkbox2")
    with col3:
        checkbox3 = st.checkbox("Select this item", key="checkbox3")

    # Check if any of the checkboxes is selected
    if checkbox1 or checkbox2 or checkbox3:
        st.write("You have successfully enrolled in the competition.")
        st.write("The winners will be announced on December 12th.")