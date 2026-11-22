import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

SCALE_RANGE = (0.5, 2.0)
rng = np.random.default_rng(0)
tf.random.set_seed(0)

def center(L):
    return L - L.mean(axis=1, keepdims=True)

def to_log(X):
    return center(np.log10(np.maximum(X, 1e-30)))

def augment(L):
    noise = np.log10(np.clip(1 + rng.normal(0, 0.03, L.shape), 0.05, None))
    scale = rng.uniform(np.log10(SCALE_RANGE[0]), np.log10(SCALE_RANGE[1]), (len(L), 1))
    return center(L + noise + scale)

train_df = pd.read_excel("particle_dataset_all_materials.xlsx")
bad = (train_df["Shape"] == "Cube") & train_df["Radius_X(nm)"].isin([30, 45])
train_df = train_df[~bad]

le = LabelEncoder()
y_all = le.fit_transform(train_df["Material"].astype(str).values).astype("int32")
L_all = to_log(train_df.iloc[:, :500].values.astype(np.float64))

L_tr, L_val, y_tr, y_val = train_test_split(L_all, y_all, test_size=0.15, stratify=y_all, random_state=0)
X_tr = np.vstack([L_tr] + [augment(L_tr) for _ in range(8)])
Y_tr = np.concatenate([y_tr] * (8 + 1))
X_val = augment(L_val)

scaler = StandardScaler().fit(X_tr)
p = rng.permutation(len(X_tr))
X_tr, Y_tr = scaler.transform(X_tr)[p].astype("float32"), Y_tr[p]
X_val = scaler.transform(X_val).astype("float32")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

cnn = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((500, 1), input_shape=(500,)),
    
    tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    
    tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(5, activation='softmax')
])

cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.fit(X_tr, Y_tr, validation_data=(X_val, y_val), epochs=200, batch_size=64,
      callbacks=[tf.keras.callbacks.EarlyStopping(
          "val_loss", patience=10, restore_best_weights=True)])
cnn.save('cnn.keras')


mlp = tf.keras.Sequential([
        tf.keras.Input((500,)),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(len(le.classes_), activation="softmax")
])

mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
mlp.fit(X_tr, Y_tr, validation_data=(X_val, y_val), epochs=200, batch_size=64,
      callbacks=[tf.keras.callbacks.EarlyStopping(
          "val_loss", patience=10, restore_best_weights=True)])
mlp.save('mlp.keras')

inp = tf.keras.Input((500,))
out = tf.keras.layers.Average()([cnn(inp), mlp(inp)])
ensemble = tf.keras.Model(inp, out, name="ensemble")
ensemble.save("model.keras")
