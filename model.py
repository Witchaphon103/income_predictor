import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib

# โหลด Dataset ใหม่จาก dataset.csv
data = pd.read_csv('data/dataset.csv')

# เลือก Feature และ Target Variable สำหรับการทำนายรายได้
X = data[['Age', 'Gender', 'Education', 'Occupation', 'Relationship Status', 'Marital Status']]
y = data['Income']

# One-hot encoding สำหรับ categorical variables
X = pd.get_dummies(X)

# แบ่งข้อมูลเป็น Training และ Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึก Decision Tree Model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# บันทึกโมเดลเป็น income_predictor_model.pkl
joblib.dump(model, 'model/income_predictor_model.pkl')