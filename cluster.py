import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib 

def calculate_rfm(sales_data):
    # Convert 'order_date' field to datetime if necessary
    sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])

    analysis_date = sales_data['order_date'].max() + pd.DateOffset(1)

    # Calculate RFM
    rfm = sales_data.groupby('User Name').agg({
        'order_date': lambda x: (analysis_date - x.max()).days,  # Recency
        'order_id': 'nunique',                                   # Frequency
        'total': 'sum'                                           # Monetary
    }).reset_index()

    rfm.columns = ['User Name', 'Recency', 'Frequency', 'Monetary']

    # Standardize the RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Summarize the clusters
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'User Name': 'count'  # Number of customers in each cluster
    }).reset_index()

    cluster_summary.columns = ['Cluster', 'Mean Recency', 'Mean Frequency', 'Mean Monetary', 'Customer Count']

    # Add segment names
    conditions = [
        (rfm['Cluster'] == 0),
        (rfm['Cluster'] == 1),
        (rfm['Cluster'] == 2)
    ]
    segment_names = ['Low Value', 'Medium Value', 'High Value']
    rfm['Segment'] = np.select(conditions, segment_names, default='Unknown')

    return rfm, cluster_summary
