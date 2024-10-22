from flask import Flask, render_template, request
from math import ceil
from cluster import calculate_rfm  

app = Flask(__name__)

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route hiển thị danh sách khách hàng với bộ lọc phân khúc
@app.route('/function/segmentation_of_customers', methods=['GET', 'POST'])
def list_customers():
    rfm_data, cluster_summary = calculate_rfm()

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
    rfm_data, cluster_summary = calculate_rfm()

    return render_template('Chart_segment_customers.html', 
                           rfm_data=rfm_data.to_dict(orient='records'),
                           cluster_summary=cluster_summary.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
