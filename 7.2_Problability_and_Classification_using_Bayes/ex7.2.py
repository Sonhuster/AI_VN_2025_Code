import numpy as np

# Day = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
Outlook = ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Overcast', 'Sunny', 'Rain']
Temperature = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild']
Humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal']
Wind = ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak']
PlayTennis = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']

Day = ['Weekday', 'Weekday', 'Weekday', 'Holiday', 'Saturday', 'Weekday', 'Holiday', 'Sunday', 'Weekday', 'Weekday', 'Saturday', 'Weekday', 'Weekday', 'Weekday', 'Weekday', 'Saturday', 'Weekday', 'Holiday', 'Weekday', 'Weekday']
Season = ['Spring', 'Winter', 'Winter', 'Winter', 'Summer', 'Autumn', 'Summer', 'Summer', 'Winter', 'Summer', 'Spring', 'Summer', 'Winter', 'Summer', 'Winter', 'Autumn', 'Autumn', 'Spring', 'Spring', 'Spring']
Fog = ['None', 'None', 'None', 'High', 'Normal', 'Normal', 'High', 'Normal', 'High', 'None', 'High', 'High', 'Normal', 'High', 'Normal', 'High', 'None', 'Normal', 'Normal', 'Normal']
Rain = ['None', 'Slight', 'None', 'Slight', 'None', 'None', 'Slight', 'None', 'Heavy', 'Slight', 'Heavy', 'Slight', 'None', 'None', 'Heavy', 'Slight', 'Heavy', 'Slight', 'None', 'Heavy']
Class = ['On Time', 'On Time', 'On Time', 'Late', 'On Time', 'Very Late', 'On Time', 'On Time', 'Very Late', 'On Time', 'Cancelled', 'On Time', 'Late', 'On Time', 'Vary Late', 'On Time', 'On Time', 'On Time', 'On Time', 'On Time']

def create_training_data(class_number: int=0):
    """Create the training dataset for tennis prediction."""
    if class_number == 0:
        data = np.vstack((Outlook, Temperature, Humidity, Wind, PlayTennis)).T
    elif class_number == 1:
        data = np.vstack((Day, Season, Fog, Rain, Class)).T

    return np.array(data)

def compute_prior_probabilities(train_data):
    """
        Calculate prior probabilities P(Play Tennis = Yes/No).

        Args:
        train_data: Training dataset

        Returns:
        Array of prior probabilities [P(No), P(Yes)]
    """
    class_names = ["No", "Yes"] if class_number == 0 else ["On Time", "Late", "Very Late", "Cancelled"]
    total_samples = len(train_data)
    prior_probs = np.zeros(len(class_names))
    for outcome_idx in range(len(class_names)):
        prior_probs[outcome_idx] = np.sum(np.isin(train_data[:, -1], class_names[outcome_idx])) / total_samples
    ### Your code here
    return prior_probs

def compute_conditional_probabilities(train_data):
    """
    Calculate conditional probabilities P(Feature|Class) for all features.

    Args:
    train_data: Training dataset

    Returns:
    Tuple of (conditional_probabilities, feature_values)
    """
    class_names = ["No", "Yes"] if class_number == 0 else ["On Time", "Late", "Very Late", "Cancelled"]
    n_features = train_data.shape[1] - 1 # Exclude target column
    conditional_probs = []
    feature_values = []
    for feature_idx in range(n_features):
        # Get unique values for this feature
        unique_values = np.unique(train_data[:, feature_idx])
        feature_values.append(unique_values)
        # Initialize conditional probability matrix
        feature_cond_probs = np.zeros((len(class_names), len(unique_values)))

        for class_idx, class_name in enumerate(class_names):
            # Get samples for this class
            # Your code here

            for value_idx, value in enumerate(unique_values):
                # Count occurrences of this feature value in this class

                # Calculate conditional probability
                # Your code here
                arr_pB = np.where(np.isin(train_data[:, -1], class_name))
                arr_pA = np.where(np.isin(train_data[:, feature_idx], value))
                feature_cond_probs[class_idx, value_idx] = len(np.intersect1d(arr_pA, arr_pB)) / len(arr_pB[0])

        conditional_probs.append(feature_cond_probs)

    return conditional_probs, feature_values

def get_feature_index(feature_value, feature_values):
    """
    Get the index of a feature value in the feature values array.
    Args:
    feature_value: Value to find
    feature_values: Array of possible feature values
    Returns:
    Index of the feature value
    """
    return np.where(feature_values == feature_value)[0][0]

def train_naive_bayes(train_data):
    """
    Train the Naive Bayes classifier.
    Args:
    train_data: Training dataset
    Returns:
    Tuple of (prior_probabilities, conditional_probabilities,
    feature_names)
    """
    # Calculate prior probabilities
    prior_probabilities = compute_prior_probabilities(train_data)
    # Calculate conditional probabilities
    conditional_probabilities, feature_names = compute_conditional_probabilities(train_data)
    return prior_probabilities, conditional_probabilities, feature_names

def predict_tennis(X, prior_probabilities, conditional_probabilities, feature_names):
    """
    Make a prediction for given features.
    Args:
    X: List of feature values [Outlook, Temperature, Humidity,
    Wind]
    prior_probabilities: Prior probabilities for each class
    conditional_probabilities: Conditional probabilities for
    each feature
    feature_names: Names/values for each feature
    Returns:
    Tuple of (prediction, probabilities)
    """
    class_names = ["No", "Yes"] if class_number == 0 else ["On Time", "Late", "Very Late", "Cancelled"]
    # Get feature indices
    feature_indices = []
    for i, feature_value in enumerate(X):
        feature_indices.append(get_feature_index(feature_value, feature_names[i]))
    # Calculate probabilities for each class
    class_probabilities = []
    for class_idx in range(len(class_names)):
        prob = prior_probabilities[class_idx]
        for feature_idx, idx in enumerate(feature_indices):
            cond_prob = conditional_probabilities[feature_idx][class_idx, idx]

            prob *= cond_prob
        class_probabilities.append(prob)
        # Start with prior probability
        # Multiply by conditional probabilities
        # Your code here
        # Normalize probabilities
    total_prob = sum(class_probabilities)
    print(class_probabilities)
    if total_prob > 0:
        normalized_probs = [p / total_prob for p in
                            class_probabilities]
    else:
        normalized_probs = np.ones_like(class_names, dtype=int) / len(class_names)  # Default if all probabilities are 0
    # Make prediction
    predicted_class_idx = np.argmax(class_probabilities)
    prediction = class_names[predicted_class_idx]
    # Create probability dictionary
    prob_dict = {
        "No": round(normalized_probs[0].item(), 9),
        "Yes": round(normalized_probs[1].item(), 9)
    }
    return prediction, prob_dict

class_number = 1
train_data = create_training_data(class_number=class_number)

# X = ["Sunny","Cool", "High", "Strong"]
X = ["Weekday", "Winter", "High", "Heavy"]

prior_probs, conditional_probs, feature_names = train_naive_bayes(train_data)
print(prior_probs)
prediction, prob_dict = predict_tennis(X, prior_probs, conditional_probs, feature_names)
print(prob_dict)
if prediction == 'No':
    print("Ad should not go!")
else:
    print("Ad should go!")