import pandas as pd

# Load dữ liệu đơn hàng
orders_2009_2010 = pd.read_csv('Order_2009_2010.csv', delimiter=';')
orders_2010_2011 = pd.read_csv('Order_2010_2011.csv', delimiter=';')

def get_order_trends():
    # Phân tích xu hướng bán hàng, tính tổng số đơn hàng, tổng giá trị
    orders_2009_2010['TotalPrice'] = orders_2009_2010['Quantity'] * orders_2009_2010['Price'].apply(lambda x: float(x.replace(',', '.')))
    orders_2010_2011['TotalPrice'] = orders_2010_2011['Quantity'] * orders_2010_2011['Price'].apply(lambda x: float(x.replace(',', '.')))
    
    total_sales_2009_2010 = orders_2009_2010['TotalPrice'].sum()
    total_sales_2010_2011 = orders_2010_2011['TotalPrice'].sum()
    
    return {
        'total_sales_2009_2010': total_sales_2009_2010,
        'total_sales_2010_2011': total_sales_2010_2011
    }
