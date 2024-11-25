from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  


def prepare_data(collection):
    data = pd.DataFrame(list(collection.find()))
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')

    customer_data = data.groupby('User Name').agg(
        total_spent=('total', 'sum'),
        avg_order_value=('total', 'mean'),
        total_discount_received=('discount_amount', 'sum'),
        avg_discount_percent=('Discount_Percent', 'mean'),
        region=('Region', 'first'),
        city=('City', 'first'),
        state=('State', 'first'),
        full_name=('full_name', 'first'),   
        email=('E Mail', 'first'),           
        age=('age', 'first'),              
        gender=('Gender', 'first'),        
        unique_order_dates=('order_date', 'nunique'),
        last_order_date=('order_date', 'max')
    )

    customer_data['returned'] = customer_data['unique_order_dates'] > 1

    label_encoders = {}
    for column in ['region', 'city', 'state', 'gender']:
        le = LabelEncoder()
        customer_data[column] = le.fit_transform(customer_data[column].astype(str))
        label_encoders[column] = le

    return customer_data, label_encoders

def split_train_test(customer_data):
    customer_data['order_quarter'] = customer_data['last_order_date'].dt.to_period('Q')
    last_quarter = customer_data['order_quarter'].max()

    train_data = customer_data[customer_data['order_quarter'] < last_quarter]
    test_data = customer_data[customer_data['order_quarter'] == last_quarter]

    return train_data, test_data

def train_model(train_data):
    X_train = train_data.drop(columns=['returned', 'unique_order_dates', 'last_order_date', 'order_quarter','full_name', 'email',])
    y_train = train_data['returned']

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_dict = dict(zip(feature_names, feature_importances * 100))
    
    joblib.dump(model, 'customer_retention_model.joblib')
    joblib.dump(feature_importance_dict, 'feature_importance.joblib')
    print("Model and feature importance saved successfully.")
    return model

def evaluate_model(model, train_data, test_data, label_encoders):

    X_test = test_data.drop(columns=['returned', 'unique_order_dates', 'last_order_date', 'order_quarter', 'full_name', 'email'])
    y_test = test_data['returned']

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    total_customers_current_quarter = test_data.shape[0]
    previous_quarter_data = train_data[train_data['order_quarter'] == (test_data['order_quarter'].iloc[0] - 1)]
    total_customers_previous_quarter = previous_quarter_data.shape[0]
    
    test_data['order_quarter'] = test_data['last_order_date'].dt.to_period('Q').apply(
        lambda x: f"Quý {x.quarter} năm {x.start_time.year}"
    )
    last_quarter = test_data['order_quarter'].iloc[0]
    
    predicted_retention_rate = y_pred.mean() * 100
    predicted_churn_rate = 100 - predicted_retention_rate
    churn_customers = (y_pred == 0).sum()
    prediction_accuracy = accuracy * 100

    churn_probabilities = model.predict_proba(X_test)[:, 0] * 100
    churn_details = test_data[['full_name', 'email', 'returned']].copy()
    churn_details['churn_probability'] = churn_probabilities

    demographic_data = {
        "gender_distribution": test_data.groupby('gender')['returned'].mean() * 100,
        "age_distribution": test_data.groupby('age')['returned'].mean() * 100,
        "region_distribution": test_data.groupby('region')['returned'].mean() * 100
    }

    # Giải mã tên vùng địa lý
    region_indices = demographic_data["region_distribution"].index.tolist()
    region_names = label_encoders['region'].inverse_transform(region_indices)
    demographic_data["region_distribution"].index = region_names
    


    return {
        "total_customers_current_quarter": total_customers_current_quarter,
        "total_customers_previous_quarter": total_customers_previous_quarter,
        "last_quarter": last_quarter,  
        "predicted_churn_rate": predicted_churn_rate,
        "churn_customers": churn_customers,
        "prediction_accuracy": prediction_accuracy,
        "churn_details": churn_details.to_dict(orient="records"),
        "demographic_data": {
            "gender_distribution": demographic_data["gender_distribution"].to_dict(),
            "age_distribution": demographic_data["age_distribution"].to_dict(),
            "region_distribution": demographic_data["region_distribution"].to_dict()
        }
    }

# # Sử dụng hàm
# client = MongoClient("mongodb://localhost:27017/")
# db = client['my_database']
# collection = db['sales_06_FY2020-21']

# # Chuẩn bị và tách dữ liệu
# customer_data,label_encoders = prepare_data(collection)
# train_data, test_data = split_train_test(customer_data)

# # Huấn luyện mô hình và lưu lại
# model = train_model(train_data)

