from flask import Flask, render_template, request
from flask import redirect, url_for
from pymongo import MongoClient
from math import ceil
from cluster import calculate_rfm
from data_customer_return import prepare_data, split_train_test, train_model, evaluate_model
import pandas as pd
import joblib
import os


app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['my_database']
sales_collection = db['sales_06_FY2020-21']

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route hiển thị danh sách khách hàng với bộ lọc phân khúc
@app.route('/function/segmentation_of_customers', methods=['GET', 'POST'])
def list_customers():
    # Fetch data from MongoDB
    sales_data_cursor = sales_collection.find({})
    sales_data = pd.DataFrame(list(sales_data_cursor))

    # Pass sales data to calculate_rfm
    rfm_data, cluster_summary = calculate_rfm(sales_data)

    selected_cluster = request.args.get('cluster', None)

    if selected_cluster is not None and selected_cluster.isdigit():
        selected_cluster = int(selected_cluster)
        rfm_data = rfm_data[rfm_data['Cluster'] == selected_cluster]

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
    # Đường dẫn tới các file đã lưu
    rfm_file_path = 'rfm_data.csv'
    cluster_summary_file_path = 'cluster_summary.csv'
    kmeans_model_path = 'kmeans_model.joblib'

    # Kiểm tra nếu các file đã tồn tại
    if os.path.exists(rfm_file_path) and os.path.exists(cluster_summary_file_path) and os.path.exists(kmeans_model_path):
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
        joblib.dump(kmeans, kmeans_model_path)

    # Chuyển đổi dữ liệu thành định dạng dictionary để truyền sang template
    return render_template('Chart_segment_customers.html', 
                           rfm_data=rfm_data.to_dict(orient='records'),
                           cluster_summary=cluster_summary.to_dict(orient='records'))

@app.route('/function/Predicting_Returning_Customers')
def Predicting_Returning_Customers():
    loaded_model = joblib.load('customer_retention_model.joblib')
    feature_importance_dict = joblib.load('feature_importance.joblib')
    print("Loaded model and feature importance successfully.")

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
@app.route('/function/retrain_model', methods=['POST'])
def retrain_model():
    customer_data, label_encoders = prepare_data(sales_collection)
    train_data, _ = split_train_test(customer_data)

    model = train_model(train_data)

    return redirect(url_for('Predicting_Returning_Customers'))

if __name__ == '__main__':
    app.run(debug=True)
