import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
housing_data = pd.read_csv('housing.csv')
# Convert categorical variables to numeric values
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for feature in categorical_features:
    housing_data[feature] = housing_data[feature].map({'yes': 1, 'no': 0})

housing_data['furnishingstatus'] = housing_data['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

# Define features and target
X = housing_data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
y = housing_data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "House Price Prediction", "Projects", "About"])

if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Welcome to House Price Predictor</h1>", unsafe_allow_html=True)
    st.write("Use the sidebar to navigate through the app.")
    st.write("""
        Welcome to the **AI-Powered Attire Classifier**!  
        📸 Upload or capture an image, and the model will classify it as **Formal** or **Informal**.
        
        - Uses **OpenAI CLIP** for classification  
        - Works on **both desktop & mobile**  
        - Supports **live camera capture**  
        
        **🔍 Get Started:** Click on **"📸 Classify Image"** in the sidebar!
    """)

elif page == "House Price Prediction":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>House Price Prediction App</h1>", unsafe_allow_html=True)
    st.write("Fill in the details below to predict the house price.")

    # Layout using columns
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1500, step=100)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
        stories = st.number_input("Number of Stories", min_value=1, max_value=5, value=1)
        parking = st.number_input("Number of Parking Spaces", min_value=0, max_value=5, value=1)

    with col2:
        mainroad = st.radio("On Main Road?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        guestroom = st.radio("Guestroom Available?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        basement = st.radio("Basement Available?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        hotwaterheating = st.radio("Hot Water Heating?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        airconditioning = st.radio("Air Conditioning?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        prefarea = st.radio("Preferred Area?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        furnishingstatus = st.radio("Furnishing Status", [0, 1, 2], format_func=lambda x: ['Unfurnished', 'Semi-furnished', 'Furnished'][x])

    st.markdown("---")

    # Prediction button
    if st.button("Predict Price", use_container_width=True):
        user_input = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])
        pred_price = model.predict(user_input)[0]
        
        st.success(f"Predicted House Price: **${pred_price:,.2f}**")

    # Visualization
    st.subheader("Price Distribution based on Bedrooms")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(housing_data['bedrooms'], housing_data['price'], color='blue', alpha=0.5, label='Actual Prices')
    ax.set_xlabel('Number of Bedrooms')
    ax.set_ylabel('Price')
    ax.set_title('House Price vs Number of Bedrooms')
    ax.legend()
    st.pyplot(fig)

elif page == "Projects":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Projects</h1>", unsafe_allow_html=True)
    st.write("""
        **Features:**
        - 📸 Capture images from a camera
        - 📤 Upload an image
        - 🏆 AI-based classification with **CLIP**
        - 🌐 Mobile-friendly UI
        
        **Developed by:** Aditya Sharma  
        **GitHub:** [Click Here](https://github.com/adityasysnet)
    """)


elif page == "About":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>About</h1>", unsafe_allow_html=True)
    st.write("This application is built to predict house prices based on multiple factors using Machine Learning.")
    st.write("The model is trained on the Housing dataset which contains information about various houses and their prices.")
    st.write("""
        **Features:**
        - 📸 Capture images from a camera
        - 📤 Upload an image
        - 🏆 AI-based classification with **CLIP**
        - 🌐 Mobile-friendly UI
        
        **Developed by:** Aditya Sharma  
        **GitHub:** [Click Here](https://github.com/adityasysnet)
    """)