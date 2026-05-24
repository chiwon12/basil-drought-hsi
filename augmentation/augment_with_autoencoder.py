# AE-based spectral augmentation by latent interpolation.
#
# Usage:
#   python augment_with_autoencoder.py --sensor sw
#   python augment_with_autoencoder.py --sensor vn
#
# Default input  : data/train_test_split/train_{swir,vnir}.csv
# Default output : data/augmentation/ae_{swir,vnir}_n300.csv
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths are resolved relative to this script: git_open/augmentation/<this>
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'
SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}

parser = argparse.ArgumentParser()
parser.add_argument('--sensor', choices=['sw', 'vn'], required=True)
parser.add_argument('--target', type=int, default=300, help='target samples per class')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--input', default=None,
                    help='override input CSV (default: data/train_test_split/train_{swir,vnir}.csv)')
parser.add_argument('--output_dir', default=None,
                    help='output directory (default: data/augmentation/)')
args = parser.parse_args()

s_long = SENSOR_LONG[args.sensor]
input_path = args.input if args.input else str(DATA / 'train_test_split' / f'train_{s_long}.csv')
out_dir = args.output_dir if args.output_dir else str(DATA / 'augmentation')

os.makedirs(out_dir, exist_ok=True)
output_path = f"{out_dir}/ae_{s_long}_n{args.target}.csv"
np.random.seed(args.seed)
import tensorflow as tf

tf.random.set_seed(args.seed)

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드 및 전처리
print(f"[input ] {input_path}")
print(f"[output] {output_path}")
df = pd.read_csv(input_path)
# feat_* 컬럼만 X로, label만 y로 (sample_id, cluster_sample_id 제외)
feat_cols = [c for c in df.columns if c.startswith("feat_")]
X = df[feat_cols].values
y = df["label"].values
print(f"  rows={len(df)}, feat_cols={len(feat_cols)}, classes={dict(zip(*np.unique(y, return_counts=True)))}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]
encoding_dim = 32

# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder 정의
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation="relu")(input_layer)
encoded_output = Dense(encoding_dim, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(encoded_output)
decoded_output = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded_output)
encoder = Model(input_layer, encoded_output)

# 디코더 분리 정의
encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-2](encoded_input)
decoder_output = autoencoder.layers[-1](decoder_layer1)
decoder = Model(encoded_input, decoder_output)

# 컴파일 및 학습
autoencoder.compile(optimizer=Adam(1e-3), loss="mse")
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=64, verbose=0)

# ─────────────────────────────────────────────────────────────────────────────
# 클래스별 균등 보간 증강 수행
X_aug, y_aug = [], []

# 1. 클래스별 현재 개수 확인
class_counts = {cls: sum(y_encoded == cls) for cls in np.unique(y_encoded)}

# 2. 최대 클래스 수 기준 증강
# max_class_count = max(class_counts.values())

# 2.고정 증강 목표 수 설정
TARGET_PER_CLASS = args.target
for class_idx in np.unique(y_encoded):
    class_mask = y_encoded == class_idx
    class_X = X_scaled[class_mask]
    latent = encoder.predict(class_X)

    n_existing = len(class_X)

    n_to_generate = TARGET_PER_CLASS - n_existing

    print(f"▶ {le.inverse_transform([class_idx])[0]}: 기존 {n_existing}개 → {TARGET_PER_CLASS}개 되도록 {n_to_generate}개 증강")

    i = 0
    count = 0
    while count < n_to_generate and len(latent) >= 2:
        z1 = latent[i % len(latent)]
        z2 = latent[(i + 1) % len(latent)]
        z_interp = 0.5 * z1 + 0.5 * z2
        x_interp = decoder.predict(np.expand_dims(z_interp, axis=0))
        X_aug.append(x_interp.squeeze())
        y_aug.append(class_idx)
        count += 1
        i += 2

# ─────────────────────────────────────────────────────────────────────────────
# 기존 + 증강 데이터 합치기
X_total = np.vstack([X_scaled, np.array(X_aug)])
y_total = np.concatenate([y_encoded, np.array(y_aug)])
X_total, y_total = shuffle(X_total, y_total, random_state=42)

# 역변환 및 저장 (feat_* 컬럼만 사용; sample_id/cluster_sample_id는 제외)
X_df = pd.DataFrame(scaler.inverse_transform(X_total), columns=feat_cols)
X_df["label"] = le.inverse_transform(y_total)
X_df.to_csv(output_path, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 완료 출력
print(f"\n✅ AE 균등 보간 증강 완료: {output_path}")
print(f"👉 총 샘플 수: 원본 {len(X_scaled)} + 증강 {len(X_aug)} = {len(X_total)}")
