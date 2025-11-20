import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    # For CPU 
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC_OPS'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_seed(42)


# Load the datasets
train_file = "FINALpersonaldataset.csv"
test_file = "FINALpublicdataset.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print(f"Loaded TRAIN CSV: {train_file} -> shape: {train_df.shape}")
print(f"Loaded TEST CSV:  {test_file} -> shape: {test_df.shape}")

     
# Detect target columns
target_candidates = [
    col for col in train_df.columns
    if 'activity' in col.lower() and 'type' in col.lower()
]
if target_candidates:
    target_col = target_candidates[0]
else:
    raise ValueError("Could not find an 'Activity Type' column in your training dataset.")

print(f"Target column detected: {target_col}")

     
# Clean functions for both datastsets
def clean_df(df):
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, df.nunique() > 1]
    # This is for missing values
    df = df.ffill().bfill().fillna(0)
    return df

train_df = clean_df(train_df)
test_df = clean_df(test_df)

print(f"\nTraining dataset after cleaning: {train_df.shape}")
print(f"\nTesting dataset after cleaning: {test_df.shape}")

     
# Separate features and target
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' was dropped during cleaning in training dataset.")

if target_col not in test_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in testing dataset after cleaning.")

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

     
# Aligning columns to avoid interruptions
common_cols = sorted(list(set(X_train.columns) & set(X_test.columns)))
if not common_cols:
    raise ValueError("There are none feature columns between training and testing datasets.")

X_train = X_train[common_cols].copy()
X_test = X_test[common_cols].copy()

print(f"\nAligned training features shape: {X_train.shape}")
print(f"\nAligned testing features shape:  {X_test.shape}")

     
# Determine feature types
cat_cols = [
    col for col in X_train.columns
    if X_train[col].dtype == 'object' and X_train[col].nunique() < 20
]
num_cols = [
    col for col in X_train.columns
    if np.issubdtype(X_train[col].dtype, np.number)
]

if not num_cols and not cat_cols:
    raise ValueError("No usable feature columns found (neither numeric nor categorical).")

print(f"Numeric columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

     
# Encode target variable     
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Operate unseen labels in test set without interrupted syntaxes
known_classes = set(label_encoder.classes_)
mask_known = y_test.isin(known_classes)

if not mask_known.all():
    num_unknown = (~mask_known).sum()
    print(f"\n[Warning] {num_unknown} test samples have unseen Activity Types "
          f"and will be ignored for evaluation.\n")
    X_test = X_test[mask_known].copy()
    y_test = y_test[mask_known]

y_test_encoded = label_encoder.transform(y_test)

print(f"Encoded classes: {label_encoder.classes_}")

     
# Preprocessing for training dataset
transformers = []
if num_cols:
    transformers.append(("num", StandardScaler(), num_cols))
if cat_cols:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='drop'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed training features: {X_train_processed.shape}")
print(f"Processed testing features:  {X_test_processed.shape}")

     
# Building the model  
input_dim = X_train_processed.shape[1]
num_classes = len(np.unique(y_train_encoded))

print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")

model = models.Sequential([
    layers.Dense(128, activation='relu', input_dim=input_dim, kernel_initializer='glorot_uniform'),
    layers.Dropout(0.3, seed=42),
    layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dropout(0.2, seed=42),
    layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

     
# Training on the punlic dataset   
print("\nStarting training...\n")
history = model.fit(
    X_train_processed, y_train_encoded,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
    shuffle=False
)

print("Training completed.")

     
# Evaluation on the personal dataset
y_pred_proba = model.predict(X_test_processed, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\n")
print("EVALUATION RESULTS")
print("\n")

accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)

print("\nClassification Report:\n", "\n" + classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
print("\n")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

     
# Curves   
plt.figure(figsize=(12, 5))

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

     
# Confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='PuBuGn',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

     
# Save model, preprocessor, and label encoder
model_filename = "strava_activity_classify.keras"
model.save(model_filename)

joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\nPreprocessor and Label Encoder saved.")

print("\nCompleted all tasks successfully.")
print(f"\nFinal model accuracy: {accuracy:.3f}\n")
