import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv(r'C:\Users\Dima\Downloads\data\idle\idle-100.csv')
print("Пример данных:")
print(data.head())
print("Столбцы данных:", data.columns)

np.random.seed(0)
data['target'] = np.random.choice([0, 1, 2, 3], size=len(data))  # 4 класса активности
print("Пример данных с метками активности:")
print(data.head())


window_size = 10
stride = 5

def extract_features(data, window_size, stride):
    features = []
    labels = []


    for i in range(0, len(data) - window_size, stride):
        window = data.iloc[i:i + window_size]
        feature_dict = {}


        if len(window) < window_size:
            continue


        for column in ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']:
            feature_dict[f'{column}_mean'] = window[column].mean()
            feature_dict[f'{column}_std'] = window[column].std()
            feature_dict[f'{column}_max'] = window[column].max()
            feature_dict[f'{column}_min'] = window[column].min()


        labels.append(window['target'].mode()[0])
        features.append(feature_dict)


    if not features:
        print("Нет сгенерированных признаков. Проверьте параметры window_size и stride.")
    else:
        print(f"Количество сгенерированных признаков: {len(features)}")


    return pd.DataFrame(features), np.array(labels)


X, y = extract_features(data, window_size, stride)
print("Пример сгенерированных признаков:")
print(X.head())


if X.empty:
    print("Ошибка: сгенерированный DataFrame признаков пуст.")
else:

    print("Распределение классов в данных:", np.bincount(y))


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    print("Распределение классов в обучающей выборке:", np.bincount(y_train))



    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)


    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)


    y_pred_svm = svm_model.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, zero_division=0))

    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
