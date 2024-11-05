import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tắt các cảnh báo của TensorFlow

import pandas as pd
import numpy as np
from Remake_Dataset import Remake
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf

# Thiết lập style cho đồ thị
plt.style.use('default')
sns.set_theme()

def load_and_prepare_data():
    """Hàm tải và chuẩn bị dữ liệu"""
    print("1. Đang tải và chuẩn bị dữ liệu...")
    df = Remake()
    
    # Chọn các đặc trưng quan trọng cho việc dự đoán
    features = [
        'Road Type',                                  # Loại đường
        'Speed limit',                                # Giới hạn tốc độ
        'Light Conditions',                           # Điều kiện ánh sáng
        'Weather Conditions',                         # Điều kiện thời tiết
        'Road Surface Conditions',                    # Điều kiện bề mặt đường
        'Urban or Rural Area',                        # Khu vực đô thị hay nông thôn
        'Junction Detail',                            # Chi tiết giao lộ
        'Junction Control',                           # Kiểm soát giao lộ
        'Pedestrian Crossing-Human Control',          # Qua đường có người điều khiển
        'Pedestrian Crossing-Physical Facilities'     # Cơ sở vật chất qua đường
    ]
    target = 'Accident Severity'  # Mức độ nghiêm trọng của tai nạn
    
    # Xử lý giá trị thiếu bằng cách điền giá trị phổ biến nhất
    for col in features:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df, features, target

def encode_features(df, features, target):
    """Hàm mã hóa các đặc trưng từ dạng text sang số"""
    print("2. Đang mã hóa features...")
    X = df[features].copy()
    encoders = {}
    
    # Mã hóa từng cột đặc trưng
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        encoders[column] = le
    
    # Mã hóa biến mục tiêu
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target])
    
    return X, y, encoders, le_target

def train_random_forest(X, y):
    """Hàm huấn luyện mô hình Random Forest"""
    print("3. Huấn luyện mô hình Random Forest...")
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Khởi tạo mô hình với các tham số đã tinh chỉnh
    rf_model = RandomForestClassifier(
        n_estimators=200,      # Số cây quyết định
        max_depth=20,          # Độ sâu tối đa của cây
        min_samples_split=10,  # Số mẫu tối thiểu để phân chia nút
        min_samples_leaf=5,    # Số mẫu tối thiểu tại nút lá
        random_state=42,       # Hạt giống ngẫu nhiên
        n_jobs=-1             # Sử dụng tất cả CPU có sẵn
    )
    
    # Đánh giá chéo để kiểm tra độ ổn định của mô hình
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"\nĐộ chính xác cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Huấn luyện mô hình trên toàn bộ dữ liệu training
    rf_model.fit(X_train, y_train)
    
    return rf_model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, features):
    """Hàm đánh giá hiệu suất của mô hình"""
    print("\n4. Đánh giá mô hình...")
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    # Tính và in độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nĐộ chính xác của mô hình: {accuracy:.4f}")
    
    # In báo cáo phân loại chi tiết
    print("\nBáo cáo phân loại chi tiết:")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Ma trận nhầm lẫn')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.savefig('Model/confusion_matrix.png')
    plt.close()
    
    # Vẽ biểu đồ tầm quan trọng của các đặc trưng
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Tầm quan trọng của các đặc trưng')
    plt.savefig('Model/feature_importance.png')
    plt.close()
    
    return feature_importance

def train_neural_network(X, y):
    """Hàm huấn luyện mạng neural"""
    print("\n5. Huấn luyện Neural Network để so sánh...")
    try:
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Chuyển đổi dữ liệu sang định dạng numpy
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        
        # Mã hóa one-hot cho biến mục tiêu
        num_classes = len(np.unique(y))
        y_train_cat = np.eye(num_classes)[y_train]
        y_test_cat = np.eye(num_classes)[y_test]
        
        # Xây dựng kiến trúc mạng neural
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=X.shape[1]))  # Lớp ẩn 1
        model.add(Dense(16, activation='relu'))                        # Lớp ẩn 2
        model.add(Dense(num_classes, activation='softmax'))            # Lớp đầu ra
        
        # Biên dịch mô hình
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Huấn luyện mô hình
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=20,
            batch_size=32,
            verbose=1
        )
        
        # Vẽ đồ thị quá trình huấn luyện
        plt.figure(figsize=(12, 4))
        # Đồ thị loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Loss training')
        plt.plot(history.history['val_loss'], label='Loss validation')
        plt.title('Loss của mô hình')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Đồ thị accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Accuracy training')
        plt.plot(history.history['val_accuracy'], label='Accuracy validation')
        plt.title('Độ chính xác của mô hình')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('Model/neural_network_performance.png')
        plt.close()
        
        return model, scaler
        
    except Exception as e:
        print(f"\nLỗi khi huấn luyện Neural Network: {str(e)}")
        print("Bỏ qua phần Neural Network và tiếp tục với Random Forest...")
        return None, None

def main():
    """Hàm chính điều khiển toàn bộ quá trình"""
    # Tải và chuẩn bị dữ liệu
    df, features, target = load_and_prepare_data()
    X, y, encoders, le_target = encode_features(df, features, target)
    
    # Huấn luyện và đánh giá Random Forest
    rf_model, X_train, X_test, y_train, y_test = train_random_forest(X, y)
    feature_importance = evaluate_model(rf_model, X_test, y_test, features)
    
    # Huấn luyện Neural Network
    nn_model, scaler = train_neural_network(X, y)
    
    # Lưu các mô hình và encoder
    print("\n6. Lưu các model và encoder...")
    joblib.dump(rf_model, 'Model/random_forest_model.joblib')
    joblib.dump(encoders, 'Model/feature_encoders.joblib')
    joblib.dump(le_target, 'Model/target_encoder.joblib')
    
    if nn_model is not None:
        joblib.dump(scaler, 'Model/scaler.joblib')
        nn_model.save('Model/neural_network_model.h5')
    
    # In thông tin về các file đã lưu
    print("\nQuá trình huấn luyện hoàn tất!")
    print("Các file đã được lưu trong thư mục 'Model':")
    print("- Model Random Forest: random_forest_model.joblib")
    print("- Encoders: feature_encoders.joblib")
    print("- Target Encoder: target_encoder.joblib")
    if nn_model is not None:
        print("- Neural Network: neural_network_model.h5")
        print("- Scaler: scaler.joblib")
        print("- Đồ thị Neural Network Performance: neural_network_performance.png")
    print("- Đồ thị Confusion Matrix: confusion_matrix.png")
    print("- Đồ thị Feature Importance: feature_importance.png")

# Điểm bắt đầu của chương trình
if __name__ == "__main__":
    main()