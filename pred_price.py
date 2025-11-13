import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load and preprocess data (similar to your code)
@st.cache_data
def load_and_preprocess_data():
    house = pd.read_csv("Bengaluru_House_Data.csv")
    house.drop(["availability", 'location', 'society'], axis=1, inplace=True)
    house.dropna(inplace=True)
    house['bhk'] = house['size'].apply(lambda x: x.split(" ")[0]).astype(int)
    house.drop("size", axis=1, inplace=True)
    house = house[(house.total_sqft.str.isnumeric())]
    house.total_sqft = house.total_sqft.astype(float)
    house_encoded = pd.get_dummies(house, dtype=int)
    return house_encoded

# Train the model
@st.cache_resource
def train_model():
    house_encoded = load_and_preprocess_data()
    X = house_encoded.drop('price', axis=1)
    y = house_encoded['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4545)
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_scaled = sc.transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    poly_reg = PolynomialFeatures(degree=2)
    poly_reg.fit(X_train_scaled)
    X_train_poly = poly_reg.transform(X_train_scaled)
    X_test_poly = poly_reg.transform(X_test_scaled)
    
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)
    
    return lr, sc, poly_reg, X.columns.tolist()

# Main app
def main():
    st.title("Bengaluru House Price Prediction")
    
    # Train model (cached)
    model, scaler, poly, feature_columns = train_model()
    
    # Input fields
    st.header("Enter House Details")
    area_type = st.selectbox("Area Type", ["Super built-up  Area", "Built-up  Area", "Plot  Area", "Carpet  Area"])
    total_sqft = st.number_input("Total Sqft", min_value=100.0, max_value=10000.0, value=1000.0)
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    
    if st.button("Predict Price"):
        # Create input dictionary
        input_data = {
            'total_sqft': total_sqft,
            'bath': bath,
            'balcony': balcony,
            'bhk': bhk,
            f'area_type_{area_type}': 1
        }
        
        # Create full input vector with all columns
        input_vector = {col: 0 for col in feature_columns}
        input_vector.update(input_data)
        
        # Convert to DataFrame and then to array
        input_df = pd.DataFrame([input_vector])
        input_scaled = scaler.transform(input_df.values)
        input_poly = poly.transform(input_scaled)
        
        # Predict
        prediction = model.predict(input_poly)[0]
        
        st.success(f"Predicted Price: ₹{prediction:.2f} Lakhs")

if __name__ == "__main__":
    main()
