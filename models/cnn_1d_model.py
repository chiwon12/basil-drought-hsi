# 1D-CNN with Optuna K-Fold CV + training curves + SQLite study + EarlyStopping.
#
# Default data layout (relative to git_open/):
#   data/train_test_split/{train,test}_{swir,vnir}.csv   (raw)
#   data/augmentation/{ae,gan}_{swir,vnir}_n300.csv      (ae | gan)
#
# Pick the sensor via the SENSOR env var (sw or vn); defaults to "sw".
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

label_color_map = {
    'WW': '#1f77b4', 'MD': '#ff7f0e', 'DR': '#2ca02c', 'SD': '#d62728'
}

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
SENSOR = os.environ.get('SENSOR', 'sw')
SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}[SENSOR]

data_paths = {
    "raw": str(DATA / 'train_test_split' / f'train_{SENSOR_LONG}.csv'),
    "ae":  str(DATA / 'augmentation' / f'ae_{SENSOR_LONG}_n300.csv'),
    "gan": str(DATA / 'augmentation' / f'gan_{SENSOR_LONG}_n300.csv'),
}

test_data_path = str(DATA / 'train_test_split' / f'test_{SENSOR_LONG}.csv')
output_dir = str(REPO / 'results' / f'cnn_1d_{SENSOR_LONG}')
os.makedirs(output_dir, exist_ok=True)

def preprocess_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["sample_id", "label"], errors="ignore")
    y = df["label"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_bin = label_binarize(y_encoded, classes=np.unique(y_encoded))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_encoded, y_bin, X.columns.shape[0], le.classes_, le, scaler

def build_model(input_shape, trial):
    model = Sequential()
    model.add(Conv1D(filters=trial.suggest_categorical("filters1", [32, 64, 128]),
                     kernel_size=trial.suggest_categorical("kernel_size1", [3, 5, 7]),
                     activation='relu', input_shape=input_shape))
    if trial.suggest_categorical("batch_norm1", [True, False]):
        model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=trial.suggest_categorical("pool_size1", [2, 4])))
    model.add(Dropout(trial.suggest_float("dropout1", 0.2, 0.5)))

    model.add(Conv1D(filters=trial.suggest_categorical("filters2", [32, 64, 128]),
                     kernel_size=trial.suggest_categorical("kernel_size2", [3, 5, 7]),
                     activation='relu'))
    if trial.suggest_categorical("batch_norm2", [True, False]):
        model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=trial.suggest_categorical("pool_size2", [2, 4])))
    model.add(Dropout(trial.suggest_float("dropout2", 0.2, 0.5)))

    gap_type = trial.suggest_categorical("gap_type", ["flatten", "global"])
    if gap_type == "flatten":
        model.add(Flatten())
    else:
        model.add(GlobalAveragePooling1D())

    model.add(Dense(trial.suggest_categorical("dense", [64, 128, 256]), activation='relu'))
    model.add(Dense(4, activation='softmax'))

    opt = Adam(learning_rate=trial.suggest_float("lr", 1e-4, 1e-2, log=True))
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def evaluate(model, X_data, y_true, y_bin_data, tag, class_names, name, input_len):
    pred = model.predict(X_data.reshape(-1, input_len, 1))
    pred_label = np.argmax(pred, axis=1)
    acc = accuracy_score(y_true, pred_label)
    f1 = f1_score(y_true, pred_label, average="macro")
    auc_score = roc_auc_score(y_bin_data, pred, multi_class="ovr")

    cm = confusion_matrix(y_true, pred_label)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    sns.heatmap(cm_df, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f"{name.upper()} - {tag} Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"cm_{name}_{tag}.png"))
    plt.clf()

    plt.figure(figsize=(6, 5))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin_data[:, i], pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})", color=label_color_map.get(class_names[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name.upper()} - {tag} ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"roc_{name}_{tag}.png"))
    plt.clf()

    return {"dataset": name, "phase": tag, "acc": acc, "f1": f1, "auc": auc_score}

def save_training_curves(history, name):
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(output_dir, f"{name}_history.csv"), index=False)
    plt.plot(df['loss'], label='Train Loss')
    plt.plot(df['val_loss'], label='Val Loss')
    plt.legend(); plt.title("Loss Curve")
    plt.savefig(os.path.join(output_dir, f"{name}_loss.png"))
    plt.clf()
    plt.plot(df['accuracy'], label='Train Acc')
    plt.plot(df['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title("Accuracy Curve")
    plt.savefig(os.path.join(output_dir, f"{name}_acc.png"))
    plt.clf()

all_results = []
for name, path in data_paths.items():
    X, y, y_bin, input_len, class_names, le, scaler = preprocess_data(path)
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=["sample_id", "label"], errors="ignore")
    y_test = le.transform(test_df["label"])
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    X_test_scaled = scaler.transform(X_test)

    def objective(trial):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = build_model((input_len, 1), trial)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
            model.fit(X[train_idx].reshape(-1, input_len, 1), y_bin[train_idx],
                      validation_data=(X[val_idx].reshape(-1, input_len, 1), y_bin[val_idx]),
                      epochs=200, batch_size=batch_size, verbose=0,
                      callbacks=[early_stop])
            pred = model.predict(X[val_idx].reshape(-1, input_len, 1))
            scores.append(f1_score(np.argmax(y_bin[val_idx], axis=1), np.argmax(pred, axis=1), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", study_name=f"{name}_cnn_study",
                                storage=f"sqlite:///{output_dir}/{name}_cnn_optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=80)

    best_model = build_model((input_len, 1), study.best_trial)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    history = best_model.fit(X.reshape(-1, input_len, 1), y_bin, epochs=200,
                             validation_split=0.2, batch_size=study.best_trial.params["batch_size"],
                             callbacks=[early_stop], verbose=1)

    save_training_curves(history, name)
    all_results.append(evaluate(best_model, X, y, y_bin, "train", class_names, name, input_len))
    all_results.append(evaluate(best_model, X_test_scaled, y_test, y_test_bin, "test", class_names, name, input_len))

pd.DataFrame(all_results).to_csv(os.path.join(output_dir, "cnn_optuna_combined_results.csv"), index=False)
