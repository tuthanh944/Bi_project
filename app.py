from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/function/page-list-product')
def product():
    return render_template('page-list-product.html')

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)
