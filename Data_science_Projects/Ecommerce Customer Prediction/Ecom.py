import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
from sklearn.preprocessing  import StandardScaler
# Set page title and layout
st.set_page_config(page_title="Customer Spending Predictor", layout="wide")

# Custom CSS for background image
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://indian-retailer.s3.ap-south-1.amazonaws.com/s3fs-public/2023-12/shopping-cart-moves-speed-light-backdrop-with-balloons-gift-boxes-all-live-futuristic-atmosphere-3d-render_0%20%281%29.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-font:White
        color: white;
        background-attachment: fixed;
        font-size:25;
        font-weight:bold;
        

.metric-label {
        color: white;
        font-size: 20px;
        font-weight: bold;
    }
.metric-value{
       font-size: 18px;
       color: lightgreen;
       font-weight: bold;
    
    
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the data
df = pd.read_csv("Ecommerce_Customers (2).csv")
df = df.drop(columns=['Email','Address','Avatar'])

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the bounds for filtering
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the DataFrame
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for specified columns
for col in df.columns:
    df = remove_outliers_iqr(df, col)
    
    
scaler = StandardScaler()

columns_to_scale = ['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Data preparation
X = df[['Avg Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Main title
st.markdown(
    '<h1 style="font-size: 36px; color: white; font-family: Arial;">Customer Spending Prediction</h1>',
    unsafe_allow_html=True
)

st.markdown(
    
   """<p style="font-size: 20px; color: white;">This application predicts the *Yearly Amount Spent* by customers based on their behavior metrics. 
   Please use the sidebar to enter customer details for prediction.</p>""",
    unsafe_allow_html=True
)


# Sidebar inputs
st.sidebar.header("Customer Input")
avg_session_length = st.sidebar.number_input("Average Session Length", min_value=0.0, step=0.1)
time_on_app = st.sidebar.number_input("Time on App", min_value=0.0, step=0.1)
time_on_website = st.sidebar.number_input("Time on Website", min_value=0.0, step=0.1)
length_of_membership = st.sidebar.number_input("Length of Membership", min_value=0.0, step=0.1)


# Model performance metrics
# st.subheader("<h1 class="metric-label">Model Performance"</h1>)
st.markdown(
    '<h1 style="font-size: 36px; color: white; font-family: Arial;">Model Performance</h1>',
    unsafe_allow_html=True
)

col1, col2, col3,col4= st.columns(4)


with col1:
    
    st.markdown(
    f"""
    <div>
        <p class="metric-label">Mean Absolute Error</p>
        <p class="metric-value">{mean_absolute_error(y_test, y_pred):.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
)
    
with col2:
    st.markdown(
    f"""
    <div>
        <p class="metric-label">Mean squared Error</p>
        <p class="metric-value">{mean_squared_error(y_test, y_pred):.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
    )
    
with col3:
    
    st.markdown(
   f"""
    <div>
        <p class="metric-label">Root Mean Squared Error</p>
        <p class="metric-value">{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
    )
    

with col4:
    
    st.markdown(
    f"""
    <div>
        <p class="metric-label">R2 score</p>
        <p class="metric-value">{r2_score(y_test, y_pred):.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
    )
    
# Prediction
if st.sidebar.button("Predict"):
    new_data = [[avg_session_length, time_on_app,time_on_website, length_of_membership]]
    prediction = model.predict(new_data)
    
    st.markdown(
    f"""
    <p style="font-size: 24px; font-weight: bold;color: white; ">Prediction Result:</P>
    <p class="metric-value">Predicted Yearly Amount Spent: ${prediction[0]:.2f}</p>
    """,
    unsafe_allow_html=True
)

