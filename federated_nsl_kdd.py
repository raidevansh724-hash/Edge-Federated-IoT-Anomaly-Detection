import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# =====================================================
# STEP 1: LOAD NSL-KDD DATASET
# =====================================================

columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins',
    'logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files',
    'num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty'
]

print("Loading dataset...")
data = pd.read_csv("nsl_kdd.csv", names=columns, encoding='latin1')

# =====================================================
# STEP 2: DATA CLEANING
# =====================================================

# Convert multi-class attacks to binary
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Drop difficulty column
data.drop('difficulty', axis=1, inplace=True)

# Encode categorical features
categorical_cols = ['protocol_type','service','flag']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

print("Data preprocessing completed.")
print("Dataset shape:", data.shape)

# =====================================================
# STEP 3: SPLIT FEATURES & LABEL
# =====================================================

X = data.drop('label', axis=1)
y = data['label']

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# STEP 4: SIMULATE FEDERATED CLIENTS
# =====================================================

NUM_CLIENTS = 5
ROUNDS = 5

client_data = np.array_split(X_train, NUM_CLIENTS)
client_labels = np.array_split(y_train, NUM_CLIENTS)

print(f"\nSimulating {NUM_CLIENTS} federated clients...")

# =====================================================
# STEP 5: FEDERATED TRAINING
# =====================================================

def local_train(X_local, y_local):
    model = LogisticRegression(max_iter=300)
    model.fit(X_local, y_local)
    return model.coef_, model.intercept_

def federated_average(weights, biases):
    avg_weight = np.mean(weights, axis=0)
    avg_bias = np.mean(biases, axis=0)
    return avg_weight, avg_bias

global_weight = None
global_bias = None

for r in range(ROUNDS):
    local_weights = []
    local_biases = []

    for i in range(NUM_CLIENTS):
        w, b = local_train(client_data[i], client_labels[i])
        local_weights.append(w)
        local_biases.append(b)

    global_weight, global_bias = federated_average(local_weights, local_biases)
    print(f"Round {r+1} completed.")

# =====================================================
# STEP 6: GLOBAL MODEL EVALUATION
# =====================================================

final_model = LogisticRegression()
final_model.coef_ = global_weight
final_model.intercept_ = global_bias
final_model.classes_ = np.array([0,1])

y_pred = final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== FINAL RESULTS =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("Confusion Matrix:\n", cm)
