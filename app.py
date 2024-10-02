from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# โหลดโมเดลที่ฝึกไว้ (ต้องแน่ใจว่าโมเดลได้รับการฝึกสำหรับการทำนายรายได้)
model = joblib.load('model/income_predictor_model.pkl')

@app.route('/dataset')
def dataset():
    return render_template('tempates/Dataset.html')

# โหลดข้อมูลใหม่จาก CSV
@app.route('/')
def index():
    try:
        data = pd.read_csv('data/dataset.csv')
    except Exception as e:
        return f"Error loading dataset: {e}"

    return render_template('index.html', data=data.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict():
    # โหลดข้อมูลจาก dataset.csv เพื่อแสดงในตาราง
    data = pd.read_csv('data/dataset.csv')

    # รับข้อมูลที่ผู้ใช้กรอก
    age = int(request.form['age'])
    gender = request.form['gender']
    education = request.form['education']
    occupation = request.form['occupation']
    relationship_status = request.form['relationship_status']
    marital_status = request.form['marital_status']

    # สร้าง DataFrame สำหรับใส่ข้อมูลที่รับจากฟอร์ม
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education': [education],
        'Occupation': [occupation],
        'Relationship Status': [relationship_status],
        'Marital Status': [marital_status]
    })

    # ทำ one-hot encoding
    input_data = pd.get_dummies(input_data)

    # จัดการฟีเจอร์ที่ขาดหายไป โดยการรีอินเด็กซ์ให้ตรงกับโมเดลที่ฝึกไว้
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # ทำการทำนายรายได้
    try:
        prediction = model.predict(input_data)[0]  # ตรวจสอบการทำนาย
    except Exception as e:
        return f"Error in prediction: {e}"  # ส่งข้อความแสดงข้อผิดพลาด(ถ้ามี)

    # ส่งข้อมูลกลับไปยัง template พร้อมกับค่าต่างๆ รวมถึงข้อมูลตาราง
    return render_template('index.html',
                           prediction=round(prediction, 2),
                           selected_age=age,
                           selected_gender=gender,
                           selected_education=education,
                           selected_occupation=occupation,
                           selected_relationship_status=relationship_status,
                           selected_marital_status=marital_status,
                           data=data.to_dict(orient='records'))

# run app
if __name__ == '__main__':
    app.run(debug=True)