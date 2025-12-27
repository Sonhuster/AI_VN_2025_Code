import numpy as np

# --- 1. Dữ liệu Đầu vào ---
# Đây là dữ liệu gốc từ bảng
Outlook = ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Overcast', 'Sunny', 'Rain']
Temperature = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild']
Humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal']
Wind = ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak']
PlayTennis = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']

# Các tên lớp mục tiêu
CLASS_NAMES = ["No", "Yes"]


def create_training_data():
    """Tạo tập dữ liệu đào tạo từ các mảng đã cho."""
    # Vị trí cuối cùng là cột mục tiêu (PlayTennis)
    data = np.vstack((Outlook, Temperature, Humidity, Wind, PlayTennis)).T
    return np.array(data)


def compute_prior_probabilities(train_data):
    """
    Tính xác suất tiên nghiệm P(Play Tennis = Yes/No).
    """
    total_samples = len(train_data)
    prior_probs = np.zeros(len(CLASS_NAMES))

    # Lấy cột PlayTennis, là cột cuối cùng (index -1)
    target_column = train_data[:, -1]

    for outcome_idx, class_name in enumerate(CLASS_NAMES):
        # Đếm số lần Class_Name xuất hiện
        count = np.sum(target_column == class_name)
        prior_probs[outcome_idx] = count / total_samples

    return prior_probs


def compute_conditional_probabilities(train_data):
    """
    Tính xác suất có điều kiện P(Feature|Class) cho tất cả các feature.
    """
    n_features = train_data.shape[1] - 1  # Bỏ cột mục tiêu
    conditional_probs = []
    feature_values_list = []

    for feature_idx in range(n_features):
        # Lấy các giá trị độc nhất (unique values) cho Feature này
        unique_values = np.unique(train_data[:, feature_idx])
        feature_values_list.append(unique_values)

        feature_cond_probs = np.zeros((len(CLASS_NAMES), len(unique_values)))

        for class_idx, class_name in enumerate(CLASS_NAMES):
            # 1. Lấy tất cả các mẫu thuộc Class hiện tại
            class_samples = train_data[train_data[:, -1] == class_name]
            num_class_samples = len(class_samples)  # Số mẫu thuộc Class hiện tại (mẫu số)

            for value_idx, value in enumerate(unique_values):
                # 2. Đếm số lần Feature Value xuất hiện trong Class đó (tử số)
                count_feature_in_class = np.sum(class_samples[:, feature_idx] == value)

                # 3. Tính P(Feature Value | Class) = Count(Feature=Value & Class)/Count(Class)
                if num_class_samples > 0:
                    # Sửa lỗi logic: Tính trực tiếp tỷ lệ dựa trên số lượng mẫu trong Class
                    prob = count_feature_in_class / num_class_samples
                else:
                    prob = 0.0  # Tránh chia cho 0

                feature_cond_probs[class_idx, value_idx] = prob

        conditional_probs.append(feature_cond_probs)

    return conditional_probs, feature_values_list


def get_feature_index(feature_value, unique_feature_values):
    """Lấy chỉ mục (index) của giá trị feature trong mảng các giá trị độc nhất."""
    # unique_feature_values là mảng NumPy 1D
    # np.where trả về một tuple (array([...]),), ta cần phần tử [0][0]
    return np.where(unique_feature_values == feature_value)[0][0]


def train_naive_bayes(train_data):
    """Đào tạo bộ phân loại Naive Bayes."""
    prior_probabilities = compute_prior_probabilities(train_data)
    conditional_probabilities, feature_names = compute_conditional_probabilities(train_data)
    return prior_probabilities, conditional_probabilities, feature_names


def predict_tennis(X, prior_probabilities, conditional_probabilities, feature_names):
    """
    Dự đoán Class cho các Feature đầu vào X.
    Sử dụng công thức Naive Bayes: P(Class|X) ∝ P(Class) * P(X1|Class) * P(X2|Class) * ...
    """
    # 1. Lấy chỉ mục của các Feature đầu vào
    feature_indices = []
    for i, feature_value in enumerate(X):
        try:
            # Sửa lỗi: Sử dụng hàm get_feature_index
            idx = get_feature_index(feature_value, feature_names[i])
            feature_indices.append(idx)
        except IndexError:
            # Xử lý trường hợp giá trị feature không có trong tập training
            print(f"Warning: Feature value '{feature_value}' not found in training data for feature {i}.")
            # Để đơn giản, gán idx=0 hoặc dùng Laplace smoothing
            feature_indices.append(0)

            # 2. Tính xác suất chưa chuẩn hóa cho mỗi Class
    class_probabilities = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Sửa lỗi: Bắt đầu với Prior Probability P(Class)
        prob = prior_probabilities[class_idx]

        # Nhân với Conditional Probabilities P(Xi | Class)
        for feature_idx, idx in enumerate(feature_indices):
            # Lấy xác suất P(Feature i | Class)
            cond_prob = conditional_probabilities[feature_idx][class_idx, idx]

            # Nhân vào xác suất tích lũy (Sử dụng giả định độc lập có điều kiện)
            prob *= cond_prob

        class_probabilities.append(prob)

    # 3. Chuẩn hóa và dự đoán
    total_prob = sum(class_probabilities)

    if total_prob > 0:
        # Chuẩn hóa để P(No) + P(Yes) = 1
        normalized_probs = [p / total_prob for p in class_probabilities]
    else:
        # Trường hợp tất cả các xác suất đều là 0 (ví dụ: do giá trị feature mới)
        normalized_probs = [0.5, 0.5]

    predicted_class_idx = np.argmax(class_probabilities)
    prediction = CLASS_NAMES[predicted_class_idx]

    # 4. Tạo kết quả trả về
    prob_dict = {
        "No": round(normalized_probs[0].item(), 4),
        "Yes": round(normalized_probs[1].item(), 4)
    }
    return prediction, prob_dict


# --- Khối Chạy Chương trình ---
train_data = create_training_data()
# X là bộ Feature cần dự đoán: ['Sunny', 'Cool', 'High', 'Strong']
X = ["Sunny", "Cool", "High", "Strong"]

# Đào tạo mô hình
prior_probs, conditional_probs, feature_names = train_naive_bayes(train_data)

# Thực hiện dự đoán
prediction, prob_dict = predict_tennis(X, prior_probs, conditional_probs, feature_names)

# In kết quả
print(f"Prior Probabilities (P(No), P(Yes)): {prior_probs}")
print(f"Conditional Probabilities (P(Feature|Class)): {conditional_probs}")
print(f"Input Features (X): {X}")
print(f"Predicted Class: {prediction}")
print(f"Probabilities P(Class|X): {prob_dict}")

if prediction == 'No':
    print("\nAd should not go!")
else:
    print("\nAd should go!")