import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm():
    sales_data = pd.read_csv('./data/sales_06_FY2020-21.csv', low_memory=False)
    sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])
    analysis_date = sales_data['order_date'].max() + pd.DateOffset(1)

    # Tính toán RFM
    rfm = sales_data.groupby('User Name').agg({
        'order_date': lambda x: (analysis_date - x.max()).days,  # Recency
        'order_id': 'nunique',                                   # Frequency
        'total': 'sum'                                           # Monetary
    }).reset_index()
    
    rfm.columns = ['User Name', 'Recency', 'Frequency', 'Monetary']

    # Chuẩn hóa RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Phân cụm
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Tính trung bình RFM cho từng cụm và số lượng khách hàng
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'User Name': 'count'  # Số lượng khách hàng trong mỗi cụm
    }).reset_index()

    cluster_summary.columns = ['Cluster', 'Mean Recency', 'Mean Frequency', 'Mean Monetary', 'Customer Count']

    # Thêm thông tin phân khúc
    conditions = [
        (rfm['Cluster'] == 0), 
        (rfm['Cluster'] == 1), 
        (rfm['Cluster'] == 2)
    ]
    segment_names = ['Low Value', 'Medium Value', 'High Value']
    rfm['Segment'] = np.select(conditions, segment_names, default='Unknown')
    
    return rfm, cluster_summary
