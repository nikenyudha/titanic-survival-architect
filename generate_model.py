import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
# Ganti 'train.csv' dengan nama file data kamu yang asli
df = pd.read_csv('Titanic-Dataset.csv')

# 2. FEATURE ENGINEERING RINGKAS
# Mengambil Title
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Mengisi Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Fitur Tambahan
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
df['Fare'] = np.log1p(df['Fare']) # Log Transformation sesuai proses kita

# 3. PREPROCESSING (ENCODING)
# Pilih kolom yang akan digunakan sesuai dengan Feature Importance kamu
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'Has_Cabin']
X = df[features]
y = df['Survived']

# One-Hot Encoding
X = pd.get_dummies(X, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# 4. SCALING & TRAINING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menggunakan Parameter Terbaik (Hasil Tuning GridSearchCV kamu sebelumnya)
# Sesuaikan parameter di bawah ini jika kamu ingat angka spesifiknya
model = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)
model.fit(X_scaled, y)

# 5. SAVE MODEL & SCALER (PICKLE)
with open('best_titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Simpan daftar kolom agar app.py tidak salah urutan
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model, Scaler, dan Feature Columns berhasil dibuat!")
print(f"Jumlah Fitur: {len(X.columns)}")