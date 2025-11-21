#

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
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import joblib

     
# Set all random seeds for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # For CPU determinism
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC_OPS'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_seed(42)

     
# Sparse focal loss 
def make_sparse_focal_loss(gamma=2.0):
    """
    Sparse categorical focal loss:
    - y_true: integer class labels, shape (batch,)
    - y_pred: probabilities after softmax, shape (batch, num_classes)
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_oh * tf.math.log(y_pred_clipped)
        focal_factor = tf.pow(1.0 - y_pred_clipped, gamma)
        loss = focal_factor * cross_entropy
        loss = tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)
    return loss_fn

focal_loss = make_sparse_focal_loss(gamma=2.0)

     
# Load datasets
train_file = "FINALpersonaldataset.csv"
test_file = "FINALpublicdataset.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print(f"Loaded TRAIN CSV: {train_file} -> shape: {train_df.shape}")
print(f"Loaded TEST  CSV: {test_file} -> shape: {test_df.shape}")

     
# Detect target columns
target_candidates = [
    col for col in train_df.columns
    if "activity" in col.lower() and "type" in col.lower()
]
if target_candidates:
    target_col = target_candidates[0]
else:
    raise ValueError("Could not find an 'Activity Type' column in your training dataset.")

print(f"Target column detected: {target_col}")

     
# Cleaning (no dropping; only fill NaNs)
def clean_df(df):
    df = df.copy()
    df = df.ffill().bfill().fillna(0)
    return df

train_df = clean_df(train_df)
test_df = clean_df(test_df)

print(f"\nTraining dataset after cleaning (no rows/cols dropped): {train_df.shape}")
print(f"Testing  dataset after cleaning (no rows/cols dropped): {test_df.shape}")

     
# Separate features and target
if target_col not in train_df.columns:
    raise ValueError(
        f"Target column '{target_col}' was not found in training dataset after cleaning."
    )

if target_col not in test_df.columns:
    raise ValueError(
        f"Target column '{target_col}' was not found in testing dataset after cleaning."
    )

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

print(f"\nInitial X_train shape: {X_train.shape}, y_train length: {len(y_train)}")
print(f"Initial X_test  shape: {X_test.shape},  y_test  length: {len(y_test)}")

     
# Align feature columns between train and test
common_cols = sorted(list(set(X_train.columns) & set(X_test.columns)))
if not common_cols:
    raise ValueError("There are no common feature columns between training and testing datasets.")

X_train = X_train[common_cols].copy()
X_test = X_test[common_cols].copy()

print(f"\nAligned training features shape: {X_train.shape}")
print(f"Aligned testing  features shape: {X_test.shape}")

     
# Determine feature types
cat_cols = [
    col for col in X_train.columns
    if X_train[col].dtype == "object" and X_train[col].nunique() < 20
]
num_cols = [
    col for col in X_train.columns
    if np.issubdtype(X_train[col].dtype, np.number)
]

if not num_cols and not cat_cols:
    raise ValueError("No usable feature columns found.")

print(f"Numeric columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

     
# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

known_classes = set(label_encoder.classes_)
mask_known = y_test.isin(known_classes)

if not mask_known.all():
    num_unknown = (~mask_known).sum()
    print(
        f"\n[Warning] {num_unknown} test samples have Activity Types "
        f"that did not appear in training and will be ignored ONLY for "
        f"evaluation metrics (datasets themselves remain unchanged).\n"
    )

# Use copies only for evaluation
X_test_eval = X_test[mask_known].copy()
y_test_eval = y_test[mask_known].copy()

if len(y_test_eval) == 0:
    raise ValueError(
        "No test samples with activity types seen during training; cannot evaluate the model."
    )

y_test_encoded = label_encoder.transform(y_test_eval)

print(f"Encoded classes (from training): {label_encoder.classes_}")
print(f"y_train_encoded length: {len(y_train_encoded)}")
print(f"y_test_encoded length (for evaluation): {len(y_test_encoded)}")

print("\nTrain class distribution:")
print(pd.Series(y_train).value_counts())
print("\nTest  class distribution (full test1.csv):")
print(pd.Series(y_test).value_counts())

     
# Preprocessing (scaling & one-hot; does NOT change sample counts)
transformers = []
if num_cols:
    transformers.append(("num", StandardScaler(), num_cols))
if cat_cols:
    transformers.append(
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test_eval)

print(f"\nProcessed training features: {X_train_processed.shape}")
print(f"Processed testing  features (for eval): {X_test_processed.shape}")

     
# Class weights  
classes_unique = np.unique(y_train_encoded)
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=classes_unique,
    y=y_train_encoded,
)
class_weights = {int(c): float(w) for c, w in zip(classes_unique, class_weights_arr)}

print("\nUsing class weights (balanced):")
print(class_weights)

     
# Build the model: 1D CNN + Dense  
input_dim = X_train_processed.shape[1]
num_classes = len(classes_unique)

print(f"\nInput dimension: {input_dim}, Number of classes: {num_classes}")

l2_reg = regularizers.l2(1e-4)  # mild L2 regularization

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Reshape((input_dim, 1)),

    layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=l2_reg,
    ),
    layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=l2_reg,
    ),
    layers.GlobalAveragePooling1D(),

    layers.Dense(
        128,
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=l2_reg,
    ),
    layers.Dropout(0.3, seed=42),

    layers.Dense(
        64,
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=l2_reg,
    ),
    layers.Dropout(0.2, seed=42),

    layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="glorot_uniform",
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss,
    metrics=["accuracy"],
)

model.summary()

     
# Callbacks: EarlyStopping + ReduceLROnPlateau
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1,
)

lr_reduce = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    verbose=1,
    min_lr=1e-6,
)

     
# Training
print("\nStarting training...\n")
history = model.fit(
    X_train_processed,
    y_train_encoded,
    epochs=300,              # EarlyStopping cuts this down
    batch_size=8,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop, lr_reduce],
    verbose=1,
    shuffle=False,           
)

print("\nTraining completed.")

     
# Evaluation
     
y_pred_proba = model.predict(X_test_processed, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nEVALUATION RESULTS\n")

accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test_encoded, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0)

print(
    "\nClassification Report:\n",
    "\n"
    + classification_report(
        y_test_encoded,
        y_pred,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        zero_division=0,
    ),
)
print("\n")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

     
# Training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

     
# Confusion matrix
cm = confusion_matrix(
    y_test_encoded,
    y_pred,
    labels=np.arange(num_classes),
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="PuBuGn",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
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