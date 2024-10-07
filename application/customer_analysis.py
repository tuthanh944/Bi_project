import pandas as pd
from sklearn.cluster import KMeans

# Load dữ liệu Churn_Modeling
churn_data = pd.read_csv('Churn_Modelling.csv', delimiter=';')

def get_customer_segments():
    # Chọn các cột liên quan đến hành vi và nhân khẩu học
    features = churn_data[['Age', 'CreditScore', 'Balance', 'EstimatedSalary']]
    
    # K-Means clustering để phân nhóm khách hàng
    kmeans = KMeans(n_clusters=4)
    churn_data['Segment'] = kmeans.fit_predict(features)
    
    # Tạo báo cáo phân khúc khách hàng
    segment_summary = churn_data.groupby('Segment').mean().to_dict()
    return segment_summary
