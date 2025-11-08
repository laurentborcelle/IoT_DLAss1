import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

# Set all random seeds for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # CPU deterministic behavior
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_seed(42)

# Load dataset
csv_file = "strava_activities_dataset.csv"
df = pd.read_csv(csv_file)

print(f"Loaded CSV: {csv_file} -> shape: {df.shape}")

# Detecting target column
target_candidates = [col for col in df.columns if 'activity' in col.lower() and 'type' in col.lower()]
if target_candidates:
    target_col = target_candidates[0]
else:
    raise ValueError("Could not find an 'Activity Type' column in your dataset.")

print(f"Target column detected: {target_col}")

# Checking dataset characteristics
print(f"Dataset size: {len(df)}")
print(f"Number of classes: {len(df[target_col].unique())}")
print("Class distribution:")
print(df[target_col].value_counts())

df = df.dropna(axis=0, how='all')            # Drop empty rows
df = df.dropna(axis=1, how='all')            # Drop empty columns
df = df.loc[:, ~df.columns.duplicated()]     # Remove duplicate columns
df = df.loc[:, df.nunique() > 1]             # Drop constant columns

# Missing value handling
df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

print(f"Dataset shape after cleaning: {df.shape}")

# Separating features (X) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Identifying numeric and categorical features
cat_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() < 20]
num_cols = [col for col in X.columns if np.issubdtype(X[col].dtype, np.number)]

if not num_cols:
    raise ValueError("No numeric columns found for model training.")

print(f"Numeric columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

# Target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Encoded classes: {label_encoder.classes_}")

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder='drop'
)

X_processed = preprocessor.fit_transform(X)
print(f"Processed features shape: {X_processed.shape}")

# Split dataset with fixed random state
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Building the deep learning model
input_dim = X_train.shape[1]
num_classes = len(np.unique(y_encoded))

print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")

# Create model with fixed initialization
model = models.Sequential([
    layers.Dense(128, activation='relu', input_dim=input_dim, kernel_initializer='glorot_uniform'),
    layers.Dropout(0.3, seed=42),
    layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dropout(0.2, seed=42),
    layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')
])

# Use optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train with fixed parameters
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

print("\nStarting training...\n")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
    shuffle=False   
)

print("Training completed.")

# Evaluation
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\n")
print("EVALUATION RESULTS")
print("\n")


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizations
plt.figure(figsize=(12, 5))

# Accuracy and loss
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Save model and preprocessors 
# Why should we save them?:  Saving all three components creates a complete, 
# reusable prediction system that maintains consistency between training and inference.
model_filename = "strava_activity_classify.keras"
model.save(model_filename)

joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\nPreprocessor and Label Encoder saved.")

print("\nCompleted all tasks successfully.")
print(f"\nFinal model accuracy: {accuracy:.3f}\n")
