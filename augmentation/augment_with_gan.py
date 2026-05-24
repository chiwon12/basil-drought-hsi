# GAN-based per-class spectral augmentation.
#
# Usage:
#   python augment_with_gan.py --sensor sw
#   python augment_with_gan.py --sensor vn
#
# Default input  : data/train_test_split/train_{swir,vnir}.csv
# Default output : data/augmentation/gan_{swir,vnir}_n300.csv
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Paths relative to this script: git_open/augmentation/<this>
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'
SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}

parser = argparse.ArgumentParser()
parser.add_argument('--sensor', choices=['sw', 'vn'], required=True)
parser.add_argument('--target', type=int, default=300)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--input', default=None,
                    help='override input CSV (default: data/train_test_split/train_{swir,vnir}.csv)')
parser.add_argument('--output_dir', default=None,
                    help='output directory (default: data/augmentation/)')
args = parser.parse_args()

s_long = SENSOR_LONG[args.sensor]
input_csv = args.input if args.input else str(DATA / 'train_test_split' / f'train_{s_long}.csv')
out_dir = args.output_dir if args.output_dir else str(DATA / 'augmentation')

os.makedirs(out_dir, exist_ok=True)
save_path = f"{out_dir}/gan_{s_long}_n{args.target}.csv"
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ----------------------------
# [1] 데이터 불러오기 및 전처리
# ----------------------------
print(f"[input ] {input_csv}")
print(f"[output] {save_path}")
df = pd.read_csv(input_csv)
# feat_* 컬럼만 X로 (sample_id, cluster_sample_id 제외)
feat_cols = [c for c in df.columns if c.startswith("feat_")]
X = df[feat_cols]
y = df["label"]
print(f"  rows={len(df)}, feat_cols={len(feat_cols)}, classes={dict(zip(*np.unique(y.values, return_counts=True)))}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# [2] 전체 클래스 기준 증강 수 설정
# ----------------------------
target_num = args.target
unique_classes = df["label"].unique()
latent_dim = args.latent_dim
n_features = X_scaled.shape[1]

# ----------------------------
# [3] GAN 모델 정의 함수
# ----------------------------
def build_generator():
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=latent_dim))
    model.add(Dense(n_features, activation="linear"))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=n_features))
    model.add(Dense(1, activation="sigmoid"))
    return model

# ----------------------------
# [4] 증강 루프 시작
# ----------------------------
augmented_dfs = []

for cls in unique_classes:
    X_cls = X_scaled[df["label"] == cls]
    n_existing = X_cls.shape[0]
    n_generate = target_num - n_existing

    if n_generate <= 0:
        print(f"✅ {cls} 클래스는 {n_existing}개로 충분함 → 건너뜀")
        continue

    print(f"🚀 {cls} 클래스 증강 시작 ({n_existing} → {target_num})")

    # GAN 모델 생성
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss="binary_crossentropy", optimizer=Adam())

    # GAN 학습
    epochs = args.epochs
    batch_size = args.batch_size
    for epoch in range(epochs):
        idx = np.random.randint(0, X_cls.shape[0], batch_size)
        real = X_cls[idx]
        real_labels = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake = generator(noise, training=False).numpy()
        fake_labels = np.zeros((batch_size, 1))

        discriminator.trainable = True
        discriminator.train_on_batch(real, real_labels)
        discriminator.train_on_batch(fake, fake_labels)

        discriminator.trainable = False
        gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 500 == 0:
            print(f"    [{epoch}] 진행 중...")

    # 증강 샘플 생성
    noise = np.random.normal(0, 1, (n_generate, latent_dim))
    generated = generator(noise, training=False).numpy()
    generated_inv = scaler.inverse_transform(generated)

    df_gen = pd.DataFrame(generated_inv, columns=X.columns)
    df_gen["label"] = cls
    augmented_dfs.append(df_gen)

# ----------------------------
# [5] 원본 + 증강 데이터 결합 및 저장
# ----------------------------
df_original = df.drop(columns=["sample_id"], errors="ignore")
if augmented_dfs:
    df_aug_total = pd.concat(augmented_dfs, ignore_index=True)
    df_full = pd.concat([df_original, df_aug_total], ignore_index=True)
else:
    df_full = df_original  # 증강 없을 경우 원본만

df_full.to_csv(save_path, index=False)
print(f"\n🎉 전체 데이터 저장 완료! (원본 + 증강 포함) → {save_path}")
