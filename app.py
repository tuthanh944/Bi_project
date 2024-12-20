from flask import Flask, render_template, request, jsonify
from flask import redirect, url_for
from pymongo import MongoClient
from math import ceil
from kmeans.cluster import calculate_rfm
from randomforest.data_customer_return import prepare_data, split_train_test, train_model, evaluate_model
from RNN.predict_no_of_customer import train_rnn_model,evaluate_and_forecast_rnn,prepare_data_RNN
from randomforest.data_customer_return import prepare_data, split_train_test, train_model, evaluate_model
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os
from fuzzywuzzy import fuzz



app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['my_database']
sales_collection = db['Bi']

def get_rfm_and_cluster_data():
    # Đường dẫn tới các file đã lưu
    rfm_file_path = 'kmeans/data/rfm_data.csv'
    cluster_summary_file_path = 'kmeans/data/cluster_summary.csv'

    # Kiểm tra nếu các file đã tồn tại
    if os.path.exists(rfm_file_path) and os.path.exists(cluster_summary_file_path):
        # Tải dữ liệu RFM và cluster_summary từ file CSV
        rfm_data = pd.read_csv(rfm_file_path)
        cluster_summary = pd.read_csv(cluster_summary_file_path)
    else:
        # Nếu file chưa tồn tại, lấy dữ liệu từ MongoDB và tính toán RFM
        sales_data_cursor = sales_collection.find({})
        sales_data = pd.DataFrame(list(sales_data_cursor))

        # Thực hiện tính toán RFM và phân cụm
        rfm_data, cluster_summary = calculate_rfm(sales_data)

        # Lưu kết quả vào file
        rfm_data.to_csv(rfm_file_path, index=False)
        cluster_summary.to_csv(cluster_summary_file_path, index=False)

    return rfm_data, cluster_summary

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/function/segmentation_of_customers', methods=['GET', 'POST'])
def list_customers():
    # Lấy dữ liệu RFM và cluster_summary
    rfm_data, _ = get_rfm_and_cluster_data()

    # Bộ lọc phân khúc khách hàng
    selected_cluster = request.args.get('cluster', None)
    if selected_cluster is not None and selected_cluster.isdigit():
        selected_cluster = int(selected_cluster)
        rfm_data = rfm_data[rfm_data['Cluster'] == selected_cluster]

    # Phân trang
    page = request.args.get('page', 1, type=int)
    per_page = 30
    total = len(rfm_data)
    total_pages = ceil(total / per_page)

    start = (page - 1) * per_page
    end = start + per_page
    customers_paginated = rfm_data.iloc[start:end]

    # Đếm số lượng khách hàng trong từng phân khúc
    cluster_counts = rfm_data['Cluster'].value_counts().to_dict()

    return render_template(
        'SegmentationOfCustomers.html',
        customers=customers_paginated.to_dict(orient='records'),
        cluster_counts=cluster_counts,
        page=page,
        total_pages=total_pages,
        selected_cluster=selected_cluster
    )


# Route hiển thị biểu đồ phân khúc khách hàng
@app.route('/function/chart_of_customers')
def chart_customers():
    # Lấy dữ liệu RFM và cluster_summary
    rfm_data, cluster_summary = get_rfm_and_cluster_data()

    # Chuyển đổi dữ liệu thành định dạng dictionary để truyền sang template
    return render_template(
        'Chart_segment_customers.html',
        rfm_data=rfm_data.to_dict(orient='records'),
        cluster_summary=cluster_summary.to_dict(orient='records')
    )
    
    
loaded_model = joblib.load('randomforest/model/customer_retention_model.joblib')
feature_importance_dict = joblib.load('randomforest/model/feature_importance.joblib')
print("Loaded model and feature importance successfully.")
@app.route('/function/Predicting_Returning_Customers')
def Predicting_Returning_Customers():

    customer_data, label_encoders = prepare_data(sales_collection)
    train_data, test_data = split_train_test(customer_data)
    evaluation_results = evaluate_model(loaded_model, train_data, test_data, label_encoders)
    
    # Rounding values
    evaluation_results["predicted_churn_rate"] = round(evaluation_results["predicted_churn_rate"], 2)
    evaluation_results["prediction_accuracy"] = round(evaluation_results["prediction_accuracy"], 2)
    evaluation_results["total_customers_current_quarter"] = round(evaluation_results["total_customers_current_quarter"], 2)
    evaluation_results["total_customers_previous_quarter"] = round(evaluation_results["total_customers_previous_quarter"], 2)

    high_churn_customers = [cust for cust in evaluation_results["churn_details"] if cust["churn_probability"] > 80]
    low_churn_customers = [cust for cust in evaluation_results["churn_details"] if cust["churn_probability"] < 20]
    
    top_features = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:3])
    
    churn_details_df = pd.DataFrame(evaluation_results["churn_details"])
    churn_details_file_path = "randomforest/churn_details.csv"
    churn_details_df.to_csv(churn_details_file_path, index=False)
    
    return render_template(
        'Predicting_Returning_Customers.html',
        total_customers_current_quarter=evaluation_results["total_customers_current_quarter"],
        total_customers_previous_quarter=evaluation_results["total_customers_previous_quarter"],
        predicted_churn_rate=evaluation_results["predicted_churn_rate"],
        churn_customers=evaluation_results["churn_customers"],
        prediction_accuracy=evaluation_results["prediction_accuracy"],
        high_churn_customers=high_churn_customers,
        low_churn_customers=low_churn_customers,
        churn_details=evaluation_results["churn_details"],
        gender_distribution=evaluation_results["demographic_data"]["gender_distribution"],
        age_distribution=evaluation_results["demographic_data"]["age_distribution"],
        region_distribution=evaluation_results["demographic_data"]["region_distribution"],
        last_quarter=evaluation_results['last_quarter'],
        top_features=top_features
    )
@app.route('/function/search_customer', methods=['POST'])
def search_customer():
    try:
        churn_details = pd.read_csv("randomforest/churn_details.csv")

        query = request.json.get("query", "").lower()
        
        customers = churn_details.to_dict(orient="records")

        filtered_customers = [
            customer for customer in customers
            if fuzz.partial_ratio(query, customer["full_name"].lower()) > 90
        ]
        
        return jsonify(filtered_customers), 200

    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"error": "An error occurred during search."}), 500

@app.route('/function/retrain_model', methods=['POST'])
def retrain_model():
    customer_data, label_encoders = prepare_data(sales_collection)
    train_data, _ = split_train_test(customer_data)

    model = train_model(train_data)

    return redirect(url_for('Predicting_Returning_Customers'))
@app.route('/function/retrain_model_kmeans', methods=['POST'])
def retrain_model_kmeans():
    # Đường dẫn tới các file đã lưu
    rfm_file_path = 'kmeans/data/rfm_data.csv'
    cluster_summary_file_path = 'kmeans/data/cluster_summary.csv'
    sales_data_cursor = sales_collection.find({})
    sales_data = pd.DataFrame(list(sales_data_cursor))
    # Thực hiện tính toán RFM và phân cụm
    rfm_data, cluster_summary = calculate_rfm(sales_data)
    # Lưu kết quả vào file
    rfm_data.to_csv(rfm_file_path, index=False)
    cluster_summary.to_csv(cluster_summary_file_path, index=False)
    
    return redirect(url_for('list_customers'))
@app.route('/function/retrain_model_kmeans_1', methods=['POST'])
def retrain_model_kmeans_1():
    # Đường dẫn tới các file đã lưu
    rfm_file_path = 'kmeans/data/rfm_data.csv'
    cluster_summary_file_path = 'kmeans/data/cluster_summary.csv'
    sales_data_cursor = sales_collection.find({})
    sales_data = pd.DataFrame(list(sales_data_cursor))
    # Thực hiện tính toán RFM và phân cụm
    rfm_data, cluster_summary = calculate_rfm(sales_data)
    # Lưu kết quả vào file
    rfm_data.to_csv(rfm_file_path, index=False)
    cluster_summary.to_csv(cluster_summary_file_path, index=False)
    
    return redirect(url_for('chart_customers'))
@app.route('/function/retrain_model_rnn', methods=['POST'])
def retrain_model_rnn():
    sales_data_cursor = sales_collection.find({})
    sales_data = pd.DataFrame(list(sales_data_cursor))
    X_train, X_test, y_train, y_test, scaler, data_scaled, daily_customers=prepare_data_RNN(sales_data)
    model = train_rnn_model(X_train, y_train, X_test, y_test)
    return redirect(url_for('load_predict'))
@app.route('/function/no_of_customer', methods=['GET','POST'])
def load_predict():
    sales_data_cursor = sales_collection.find({})
    sales_data = pd.DataFrame(list(sales_data_cursor))
    model = load_model('rnn/model/rnn_model.h5')
    X_train, X_test, y_train, y_test, scaler, data_scaled, daily_customers=prepare_data_RNN(sales_data)
    comparison_df, forecast_df = evaluate_and_forecast_rnn(model, X_test, y_test, scaler, data_scaled, daily_customers)
    comparison_dict = comparison_df.to_dict(orient='records')  
    forecast_dict = forecast_df.to_dict(orient='records')
    return render_template(
    'PredictNoOfCustomers.html',
        comparison_data=comparison_dict,
        forecast_data=forecast_dict
)

if __name__ == '__main__':
    app.run(debug=True)
