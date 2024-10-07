import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dữ liệu
churn_data = pd.read_csv('Churn_Modelling.csv', delimiter=';')

def predict_churn():
    X = churn_data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember']]
    y = churn_data['Exited']

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Mô hình RandomForest để dự đoán khách hàng rời bỏ
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Dự đoán và trả về kết quả
    predictions = model.predict(X_test)
    return {
        'predictions': predictions.tolist(),
        'accuracy': model.score(X_test, y_test)
    }
