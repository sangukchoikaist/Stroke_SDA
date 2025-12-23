import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from itertools import product
import pandas as pd
import pickle
from email.mime.text import MIMEText
import smtplib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def visualize_latent_space_pca(encoder, X_src, X_tgt, n_components=2):
    """PCA를 이용한 Latent Space 시각화"""
    Z_src = encoder(X_src, training=False).numpy()
    Z_tgt = encoder(X_tgt, training=False).numpy()

    Z_all = np.concatenate([Z_src, Z_tgt], axis=0)
    labels = ['Source'] * len(Z_src) + ['Target'] * len(Z_tgt)

    pca = PCA(n_components=n_components)
    Z_pca = pca.fit_transform(Z_all)

    plt.figure(figsize=(6, 6))
    for label in ['Source', 'Target']:
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(Z_pca[idx, 0], Z_pca[idx, 1], label=label, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Space PCA Projection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_input_space_pca(X_src, X_tgt, n_components=2):
    """입력 feature space에서 PCA 시각화 (평균 또는 일부 timestep 사용)"""
    # 예: 마지막 timestep의 feature만 사용
    X_src_feat = X_src[:, -1, :]  # shape: (N, F)
    X_tgt_feat = X_tgt[:, -1, :]

    X_all = np.concatenate([X_src_feat, X_tgt_feat], axis=0)
    labels = ['Source'] * len(X_src_feat) + ['Target'] * len(X_tgt_feat)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_all)

    plt.figure(figsize=(6, 6))
    for label in ['Source', 'Target']:
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Input Feature PCA Projection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_latent_space_tsne(encoder, X_src, X_tgt, perplexity=30):
    Z_src = encoder(X_src, training=False).numpy()
    Z_tgt = encoder(X_tgt, training=False).numpy()

    Z_all = np.concatenate([Z_src, Z_tgt], axis=0)
    labels = ['Source'] * len(Z_src) + ['Target'] * len(Z_tgt)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    Z_tsne = tsne.fit_transform(Z_all)

    plt.figure(figsize=(6, 6))
    for label in ['Source', 'Target']:
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(Z_tsne[idx, 0], Z_tsne[idx, 1], label=label, alpha=0.6)
    plt.title("Latent Space t-SNE Projection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def normalize_dataset(X_train, X_val=None, X_test=None):
    """
    X: (N, T, F) 형태의 시계열 데이터 (예: [윈도우 개수, 윈도우 길이, 피처 수])
    모든 시간축을 flatten하여 feature별 평균/표준편차 계산 후 정규화

    반환값:
        - 정규화된 X_train, X_val, X_test
        - scaler 객체 (inverse_transform 등에 사용 가능)
    """
    N, T, F = X_train.shape
    scaler = StandardScaler()

    # 시계열 펼쳐서 (N*T, F) 형태로 변환
    X_train_2d = X_train.reshape(-1, F)
    scaler.fit(X_train_2d)

    X_train_norm = scaler.transform(X_train_2d).reshape(N, T, F)
    results = [X_train_norm]

    for X in [X_val, X_test]:
        if X is not None:
            X_2d = X.reshape(-1, F)
            X_norm = scaler.transform(X_2d).reshape(X.shape[0], T, F)
            results.append(X_norm)
        else:
            results.append(None)

    return (*results, scaler)


def compute_mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5, fix_sigma=1.0):
    batch_size = tf.shape(source_features)[0]
    total = tf.concat([source_features, target_features], axis=0)

    total0 = tf.expand_dims(total, axis=0)
    total1 = tf.expand_dims(total, axis=1)
    L2_distance = tf.reduce_sum((total0 - total1) ** 2, axis=-1)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / tf.cast(batch_size**2 - batch_size, tf.float32)

    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = [tf.exp(-L2_distance / bw) for bw in bandwidth_list]
    kernel_matrix = sum(kernels)

    XX = kernel_matrix[:batch_size, :batch_size]
    YY = kernel_matrix[batch_size:, batch_size:]
    XY = kernel_matrix[:batch_size, batch_size:]
    YX = kernel_matrix[batch_size:, :batch_size]

    loss = tf.reduce_mean(XX + YY - XY - YX)
    return loss

def build_encoder(input_shape, lstm_units=[128, 64], dropout=0.3, latent_dim=32):
    model = models.Sequential()
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        if i == 0:
            model.add(layers.LSTM(units, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(layers.LSTM(units, return_sequences=return_seq))  # ❌ input_shape 생략
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(latent_dim))  # latent vector
    return model, latent_dim

def build_decoder(latent_dim, output_dim=2, dense_units=[32, 16]):
    model = models.Sequential()
    model.add(layers.Input(shape=(latent_dim,)))

    for units in dense_units:
        model.add(layers.Dense(units, activation='relu'))
    
    model.add(layers.Dense(output_dim, dtype='float32'))  # 출력층 (활성화 함수 없음 또는 상황에 따라 'sigmoid'나 'softmax' 등)
    
    return model

def build_lstm_decoder(latent_dim, output_dim=2, lstm_units=[32], dropout=0.2):
    model = models.Sequential()
    model.add(layers.Reshape((1, latent_dim), input_shape=(latent_dim,)))  # latent vector → 1-step sequence
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_seq))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_dim))  # output: (cos, sin)
    return model

def train_sda_dual_decoder(X_src_train, y_src_train,
                           X_tgt_train, y_tgt_train,
                        #    X_tgt_val, y_tgt_val,
                           lambda_mmd=0.5, lambda_src=0.3, lambda_tgt=1,
                           epochs=100, batch_size=64, verbose=True,
                           patience=10, latent_dim=64, learning_rate=1e-03, encoder_units=[128, 64], decoder_units=[32]):

    input_shape = X_src_train.shape[1:]
    encoder, latent_dim = build_encoder(input_shape, lstm_units=encoder_units, latent_dim=latent_dim)
    decoder_src = build_decoder(latent_dim, dense_units=decoder_units)
    decoder_tgt = build_decoder(latent_dim, dense_units=decoder_units)
    optimizer_enc = tf.keras.optimizers.Adam(learning_rate)
    optimizer_src = tf.keras.optimizers.Adam(learning_rate)
    optimizer_tgt = tf.keras.optimizers.Adam(learning_rate)
    train_losses, val_losses, src_losses, tgt_losses, mmd_losses = [], [], [], [], []
    best_val_loss = float('inf')
    wait = 0

    # Early stopping: 가중치 저장용
    best_encoder_weights = encoder.get_weights()
    best_decoder_tgt_weights = decoder_tgt.get_weights()
    best_decoder_src_weights = decoder_src.get_weights()

    for epoch in range(epochs):
        idx_src = np.random.permutation(len(X_src_train))
        idx_tgt = np.random.permutation(len(X_tgt_train))
        n_batches = min(len(idx_src), len(idx_tgt)) // batch_size
        epoch_loss = 0
        epoch_loss_mmd = 0
        epoch_loss_src = 0
        epoch_loss_tgt = 0
        for i in range(n_batches):
            b_src = idx_src[i * batch_size:(i + 1) * batch_size]
            b_tgt = idx_tgt[i * batch_size:(i + 1) * batch_size]
            Xs, ys = X_src_train[b_src], y_src_train[b_src]
            Xt, yt = X_tgt_train[b_tgt], y_tgt_train[b_tgt]

            with tf.GradientTape(persistent=True) as tape:
                zs = encoder(Xs, training=True)
                zt = encoder(Xt, training=True)
                ys_pred = decoder_src(zs, training=True)
                yt_pred = decoder_tgt(zt, training=True)
                loss_src = tf.reduce_mean(tf.keras.losses.mse(ys, ys_pred))
                loss_tgt = tf.reduce_mean(tf.keras.losses.mse(yt, yt_pred))
                mmd = compute_mmd_loss(zs, zt)
                loss = lambda_tgt * loss_tgt + lambda_mmd * mmd + lambda_src * loss_src

            grads_enc = tape.gradient(loss, encoder.trainable_variables)
            grads_src = tape.gradient(loss_src, decoder_src.trainable_variables)
            grads_tgt = tape.gradient(loss_tgt, decoder_tgt.trainable_variables)
            optimizer_enc.apply_gradients(zip(grads_enc, encoder.trainable_variables))
            optimizer_src.apply_gradients(zip(grads_src, decoder_src.trainable_variables))
            optimizer_tgt.apply_gradients(zip(grads_tgt, decoder_tgt.trainable_variables))
            epoch_loss += loss.numpy()
            epoch_loss_src += lambda_src * loss_src.numpy()
            epoch_loss_tgt += loss_tgt.numpy()
            epoch_loss_mmd += lambda_mmd * mmd.numpy()

        # 🔍 Validation
        # z_val = encoder(X_tgt_val, training=False)
        # y_val_pred = decoder_tgt(z_val, training=False)
        # val_loss = np.mean((y_tgt_val - y_val_pred.numpy()) ** 2)
        train_losses.append(epoch_loss / n_batches)
        src_losses.append(epoch_loss_src / n_batches)
        tgt_losses.append(epoch_loss_tgt / n_batches)
        mmd_losses.append(epoch_loss_mmd / n_batches)
        train_loss = epoch_loss / n_batches
        val_losses = train_losses
        # val_losses.append(val_loss)

        if verbose:
            print(f"[{epoch+1:02d}] Train loss: {train_losses[-1]:.4f} | Val loss: {val_losses[-1]:.4f}")

        # 🔁 Early stopping check
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     wait = 0
        #     best_encoder_weights = encoder.get_weights()
        #     best_decoder_tgt_weights = decoder_tgt.get_weights()
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            wait = 0
            best_encoder_weights = encoder.get_weights()
            best_decoder_tgt_weights = decoder_tgt.get_weights()
            best_decoder_src_weights = decoder_src.get_weights()
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # 가중치 복원
    encoder.set_weights(best_encoder_weights)
    decoder_tgt.set_weights(best_decoder_tgt_weights)
    decoder_src.set_weights(best_decoder_src_weights)

    return encoder, decoder_src, decoder_tgt, train_losses, val_losses, src_losses, tgt_losses, mmd_losses

def train_sda_dual_decoder_SO(X_src_train, y_src_train,
                           X_tgt_train, y_tgt_train,
                        #    X_tgt_val, y_tgt_val,
                           lambda_mmd=0.5, lambda_src=0.3, lambda_tgt=1,
                           epochs=100, batch_size=64, verbose=True,
                           patience=10, latent_dim=64, learning_rate=1e-03):

    input_shape = X_src_train.shape[1:]
    encoder, latent_dim = build_encoder(input_shape, lstm_units=[128, 64], latent_dim=latent_dim)
    decoder_src = build_decoder(latent_dim)
    decoder_tgt = build_decoder(latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_losses, val_losses, src_losses, tgt_losses, mmd_losses = [], [], [], [], []
    best_val_loss = float('inf')
    wait = 0

    # Early stopping: 가중치 저장용
    best_encoder_weights = encoder.get_weights()
    best_decoder_tgt_weights = decoder_tgt.get_weights()
    best_decoder_src_weights = decoder_src.get_weights()

    for epoch in range(epochs):
        idx_src = np.random.permutation(len(X_src_train))
        idx_tgt = np.random.permutation(len(X_tgt_train))
        n_batches = min(len(idx_src), len(idx_tgt)) // batch_size
        epoch_loss = 0
        epoch_loss_mmd = 0
        epoch_loss_src = 0
        epoch_loss_tgt = 0
        for i in range(n_batches):
            b_src = idx_src[i * batch_size:(i + 1) * batch_size]
            b_tgt = idx_tgt[i * batch_size:(i + 1) * batch_size]
            Xs, ys = X_src_train[b_src], y_src_train[b_src]
            Xt, yt = X_tgt_train[b_tgt], y_tgt_train[b_tgt]

            with tf.GradientTape(persistent=True) as tape:
                zs = encoder(Xs, training=True)
                zt = encoder(Xt, training=True)
                ys_pred = decoder_src(zs, training=True)
                yt_pred = decoder_tgt(zt, training=True)
                loss_src = tf.reduce_mean(tf.keras.losses.mse(ys, ys_pred))
                loss_tgt = tf.reduce_mean(tf.keras.losses.mse(yt, yt_pred))
                mmd = compute_mmd_loss(zs, zt)
                loss = loss_src

            total_vars = encoder.trainable_variables + decoder_src.trainable_variables
            grads = tape.gradient(loss, total_vars)
            optimizer.apply_gradients(zip(grads, total_vars))
            epoch_loss += loss.numpy()
            epoch_loss_src += lambda_src * loss_src.numpy()
            epoch_loss_tgt += loss_tgt.numpy()
            epoch_loss_mmd += lambda_mmd * mmd.numpy()

        # 🔍 Validation
        # z_val = encoder(X_tgt_val, training=False)
        # y_val_pred = decoder_tgt(z_val, training=False)
        # val_loss = np.mean((y_tgt_val - y_val_pred.numpy()) ** 2)
        train_losses.append(epoch_loss / n_batches)
        src_losses.append(epoch_loss_src / n_batches)
        tgt_losses.append(epoch_loss_tgt / n_batches)
        mmd_losses.append(epoch_loss_mmd / n_batches)
        train_loss = epoch_loss / n_batches
        val_losses = train_losses
        # val_losses.append(val_loss)

        if verbose:
            print(f"[{epoch+1:02d}] Train loss: {train_losses[-1]:.4f} | Val loss: {val_losses[-1]:.4f}")

        # 🔁 Early stopping check
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     wait = 0
        #     best_encoder_weights = encoder.get_weights()
        #     best_decoder_tgt_weights = decoder_tgt.get_weights()
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            wait = 0
            best_encoder_weights = encoder.get_weights()
            best_decoder_tgt_weights = decoder_tgt.get_weights()
            best_decoder_src_weights = decoder_src.get_weights()
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # 가중치 복원
    encoder.set_weights(best_encoder_weights)
    decoder_tgt.set_weights(best_decoder_tgt_weights)
    decoder_src.set_weights(best_decoder_src_weights)

    return encoder, decoder_src, decoder_tgt, train_losses, val_losses, src_losses, tgt_losses, mmd_losses

def train_sda_dual_decoder_TO(X_src_train, y_src_train,
                           X_tgt_train, y_tgt_train,
                        #    X_tgt_val, y_tgt_val,
                           lambda_mmd=0.5, lambda_src=0.3, lambda_tgt=1,
                           epochs=100, batch_size=64, verbose=True,
                           patience=10, latent_dim=64, learning_rate=1e-03):

    input_shape = X_src_train.shape[1:]
    encoder, latent_dim = build_encoder(input_shape, lstm_units=[128, 64], latent_dim=latent_dim)
    decoder_src = build_decoder(latent_dim)
    decoder_tgt = build_decoder(latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_losses, val_losses, src_losses, tgt_losses, mmd_losses = [], [], [], [], []
    best_val_loss = float('inf')
    wait = 0

    # Early stopping: 가중치 저장용
    best_encoder_weights = encoder.get_weights()
    best_decoder_tgt_weights = decoder_tgt.get_weights()
    best_decoder_src_weights = decoder_src.get_weights()

    for epoch in range(epochs):
        idx_src = np.random.permutation(len(X_src_train))
        idx_tgt = np.random.permutation(len(X_tgt_train))
        n_batches = min(len(idx_src), len(idx_tgt)) // batch_size
        epoch_loss = 0
        epoch_loss_mmd = 0
        epoch_loss_src = 0
        epoch_loss_tgt = 0
        for i in range(n_batches):
            b_src = idx_src[i * batch_size:(i + 1) * batch_size]
            b_tgt = idx_tgt[i * batch_size:(i + 1) * batch_size]
            Xs, ys = X_src_train[b_src], y_src_train[b_src]
            Xt, yt = X_tgt_train[b_tgt], y_tgt_train[b_tgt]

            with tf.GradientTape(persistent=True) as tape:
                zs = encoder(Xs, training=True)
                zt = encoder(Xt, training=True)
                ys_pred = decoder_src(zs, training=True)
                yt_pred = decoder_tgt(zt, training=True)
                loss_src = tf.reduce_mean(tf.keras.losses.mse(ys, ys_pred))
                loss_tgt = tf.reduce_mean(tf.keras.losses.mse(yt, yt_pred))
                mmd = compute_mmd_loss(zs, zt)
                loss = loss_tgt

            total_vars = encoder.trainable_variables + decoder_tgt.trainable_variables
            grads = tape.gradient(loss, total_vars)
            optimizer.apply_gradients(zip(grads, total_vars))
            epoch_loss += loss.numpy()
            epoch_loss_src += lambda_src * loss_src.numpy()
            epoch_loss_tgt += loss_tgt.numpy()
            epoch_loss_mmd += lambda_mmd * mmd.numpy()

        # 🔍 Validation
        # z_val = encoder(X_tgt_val, training=False)
        # y_val_pred = decoder_tgt(z_val, training=False)
        # val_loss = np.mean((y_tgt_val - y_val_pred.numpy()) ** 2)
        train_losses.append(epoch_loss / n_batches)
        src_losses.append(epoch_loss_src / n_batches)
        tgt_losses.append(epoch_loss_tgt / n_batches)
        mmd_losses.append(epoch_loss_mmd / n_batches)
        train_loss = epoch_loss / n_batches
        val_losses = train_losses
        # val_losses.append(val_loss)

        if verbose:
            print(f"[{epoch+1:02d}] Train loss: {train_losses[-1]:.4f} | Val loss: {val_losses[-1]:.4f}")

        # 🔁 Early stopping check
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     wait = 0
        #     best_encoder_weights = encoder.get_weights()
        #     best_decoder_tgt_weights = decoder_tgt.get_weights()
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            wait = 0
            best_encoder_weights = encoder.get_weights()
            best_decoder_tgt_weights = decoder_tgt.get_weights()
            best_decoder_src_weights = decoder_src.get_weights()
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # 가중치 복원
    encoder.set_weights(best_encoder_weights)
    decoder_tgt.set_weights(best_decoder_tgt_weights)
    decoder_src.set_weights(best_decoder_src_weights)

    return encoder, decoder_src, decoder_tgt, train_losses, val_losses, src_losses, tgt_losses, mmd_losses

def train_combined_network(X_src_train, y_src_train,
                           X_tgt_train, y_tgt_train,
                        #    X_tgt_val, y_tgt_val,
                           lambda_mmd=0.5, lambda_src=0.3, lambda_tgt=1,
                           epochs=100, batch_size=64, verbose=True,
                           patience=10, latent_dim=64, learning_rate=1e-03, encoder_units=[128, 64], decoder_units=[32]):

    input_shape = X_src_train.shape[1:]
    encoder, latent_dim = build_encoder(input_shape, lstm_units=encoder_units, latent_dim=latent_dim)
    decoder_tgt = build_decoder(latent_dim, output_dim=2, dense_units=decoder_units)
    model_combined = tf.keras.Sequential()
    for layer in encoder.layers:
        model_combined.add(layer)
    for layer in decoder_tgt.layers:
        model_combined.add(layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_losses = []
    best_loss = float('inf')
    wait = 0

    # Early stopping: 가중치 저장용
    best_model_weights = model_combined.get_weights()

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_tgt_train))
        n_batches = len(idx) // batch_size
        epoch_loss = 0

        for i in range(n_batches):
            b = idx[i * batch_size:(i + 1) * batch_size]
            Xt, yt = X_tgt_train[b], y_tgt_train[b]

            with tf.GradientTape() as tape:
                y_pred = model_combined(Xt, training=True)
                loss = tf.reduce_mean(tf.keras.losses.mse(yt, y_pred))

            grads = tape.gradient(loss, model_combined.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_combined.trainable_variables))
            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        if verbose:
            print(f"[{epoch+1:02d}] Train Loss: {avg_loss:.4f}")

        # Early stopping (train loss 기준)
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            best_model_weights = model_combined.get_weights()
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # 가중치 복원
    model_combined.set_weights(best_model_weights)
    n_encoder_layers = len(encoder.layers)  # 저장해 둔 encoder 사용
    encoder_new, decoder_new = split_combined_model(model_combined, n_encoder_layers, latent_dim=latent_dim)
    return encoder_new, decoder_new, decoder_new, train_losses, train_losses, train_losses, train_losses, train_losses

# 학습 후: encoder와 decoder_tgt 분리하기
def split_combined_model(model_combined, n_encoder_layers, latent_dim, output_dim=2):
    encoder_model = tf.keras.Sequential()
    decoder_model = tf.keras.Sequential()

    for layer in model_combined.layers[:n_encoder_layers]:
        encoder_model.add(layer)

    for layer in model_combined.layers[n_encoder_layers:]:
        decoder_model.add(layer)

    # Output 확인용 설정
    encoder_model.build(input_shape=(None, *model_combined.input_shape[1:]))
    decoder_model.build(input_shape=(None, latent_dim))
    return encoder_model, decoder_model


def evaluate_on_testset(encoder, decoder, X_test, y_test, name="Test"):
    z_test = encoder(X_test, training=False)
    y_pred = decoder(z_test, training=False)
    mse = tf.reduce_mean(tf.keras.losses.mse(y_test, y_pred)).numpy()
    print(f"📊 {name} MSE: {mse:.4f}")
    return y_pred.numpy(), mse

def load_stroke_h5_grouped_by_subject_and_trial(file_path, window_size=50, stride=1, pfx = 'paretic'):
    """
    Load stroke dataset from H5 file and organize it as:
    {
        'S001': {
            'T001': {'X': ..., 'y': ..., 'gc': ...},
            'T002': ...
        },
        ...
    }
    """
    subject_dict = {}
    with h5py.File(file_path, 'r') as f:
        for trial_name in f:
            grp = f[trial_name]

            # Extract subject ID and trial ID
            parts = trial_name.split('_')
            subject_id = parts[0]  # e.g., S001
            trial_id = parts[1]    # e.g., T001

            # Determine paretic side
            paretic_side = str(grp['paretic_side'][()].decode('utf-8')) \
                if isinstance(grp['paretic_side'][()], bytes) \
                else str(grp['paretic_side'][()])

            # Assign key prefix based on paretic side
            
            def get_data(prefix, key):
                return np.array(grp[f"{prefix}_{key}"]).squeeze()
            # 샘플링 간격
            dt = 0.01  # 100Hz

            theta = get_data(pfx, 'theta_est')
            
            # 미분 계산
            theta_dot = np.gradient(theta, dt)  # shape: (T,)
            # Inputs
            X = np.stack([
                get_data(pfx, 'acc_x'),
                get_data(pfx, 'acc_y'),
                get_data(pfx, 'acc_z'),
                get_data(pfx, 'gyr_x'),
                get_data(pfx, 'gyr_y'),
                get_data(pfx, 'gyr_z'),
                get_data(pfx, 'theta_est'),
                theta_dot,
            ], axis=1)

            # Gait phase
            phase_raw = np.array(grp['gc_hs']).squeeze()
            y = np.stack([np.cos(2 * np.pi * phase_raw), np.sin(2 * np.pi * phase_raw)], axis=1)

            # Windowing
            if len(X) < window_size:
                continue

            X_list, y_list, gc_list = [], [], []
            for start in range(0, len(X) - window_size + 1, stride):
                x_win = X[start:start + window_size]
                y_label = y[start + window_size - 1]
                gc_val = phase_raw[start + window_size - 1]
                X_list.append(x_win)
                y_list.append(y_label)
                gc_list.append(gc_val)

            if X_list:
                if subject_id not in subject_dict:
                    subject_dict[subject_id] = {}
                subject_dict[subject_id][trial_id] = {
                    'X': np.array(X_list),
                    'y': np.array(y_list),
                    'gc': np.array(gc_list)
                }

    return subject_dict

def build_stroke_dataset_per_subject(subject_trial_dict, val_ratio=0, test_ratio=0.1, seed=42):
    """
    각 피험자(subject)별로 trial을 나누고, 그 안에서 train/val/test 분할
    """
    np.random.seed(seed)
    result = {}

    for subject_id, trials in subject_trial_dict.items():
        trial_ids = list(trials.keys())
        np.random.shuffle(trial_ids)

        N = len(trial_ids)
        n_val = max(int(N * val_ratio), 0) if N >= 3 else 0
        n_test = max(int(N * test_ratio), 1) if N >= 3 else 0
        n_train = N - n_val - n_test
        print(f"{N}, {n_val}, {n_test}, {n_train}")

        if n_train < 1:
            print(f"⚠️ Skipping subject {subject_id} (not enough trials)")
            continue

        train_ids = trial_ids[:n_train]
        val_ids = trial_ids[n_train:n_train + n_val]
        test_ids = trial_ids[n_train + n_val:]

        def merge(ids):
            Xs, ys = [], []
            for tid in ids:
                X_data = trials[tid].get('X', None)
                y_data = trials[tid].get('y', None)
                if X_data is not None and len(X_data) > 0:
                    Xs.append(X_data)
                    ys.append(y_data)
            if len(Xs) == 0:
                return np.empty((0, 50, 7)), np.empty((0, 2))  # or return None
            return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

        result[subject_id] = {
            'train': merge(train_ids),
            'val': merge(val_ids),
            'test': merge(test_ids)
        }

    return result

# ---------- STEP 1: Load and preprocess ----------
def load_h5_to_trial_dict_with_squeeze(file_path, window_size=50, stride=5, max_walking_speed = 0.7, gc_type='gcR_hs'):
    trial_dict = {}
    with h5py.File(file_path, 'r') as f:
        for trial_name in f.keys():
            grp = f[trial_name]
            keys = list(grp.keys())

            # 보행 속도 확인
            if 'walking_speed' not in keys:
                continue
            walking_speed = np.mean(np.array(grp['walking_speed']).squeeze())
            if np.isscalar(walking_speed):
                speed_val = walking_speed
            else:
                speed_val = walking_speed[0]  # 혹시 리스트여도 첫 값 사용

            if speed_val > max_walking_speed:
                print(f"average walking speed is {walking_speed}")
                continue  # 보행 속도가 너무 빠르면 제외

            if all(k in keys for k in ['thigh_acc_x', 'thigh_acc_y', 'thigh_acc_z',
                                       'thigh_gyr_x', 'thigh_gyr_y', 'thigh_gyr_z',
                                       'theta_est', gc_type]):
                def get_data(key):
                    return np.array(grp[key]).squeeze()
                # 샘플링 간격
                dt = 0.01  # 100Hz

                # theta_est 불러오기
                theta = get_data('theta_est')  # shape: (T,)

                # 미분 계산
                theta_dot = np.gradient(theta, dt)  # shape: (T,)
                X = np.stack([
                    get_data('thigh_acc_x'),
                    get_data('thigh_acc_y'),
                    get_data('thigh_acc_z'),
                    get_data('thigh_gyr_x'),
                    get_data('thigh_gyr_y'),
                    get_data('thigh_gyr_z'),
                    theta,
                    theta_dot,
                ], axis=1)

                phase = get_data(gc_type) * 2 * np.pi * 0.01
                y = np.stack([np.cos(phase), np.sin(phase)], axis=1)

                if len(X) < window_size:
                    continue

                X_list, y_list = [], []
                for start in range(0, len(X) - window_size + 1, stride):
                    x_win = X[start:start + window_size]
                    y_label = y[start + window_size - 1]
                    X_list.append(x_win)
                    y_list.append(y_label)

                if X_list:
                    trial_dict[trial_name] = {
                        # 'X': np.array(X_list),
                        # 'y': np.array(y_list)
                        'X': np.stack(X_list).astype(np.float32),
                        'y': np.stack(y_list).astype(np.float32)
                    }
    return trial_dict

def split_trials_by_random(trial_dict, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    trial_dict의 키(trial 이름)를 기준으로 trial 단위로 무작위 분할
    """
    all_keys = list(trial_dict.keys())
    random.seed(seed)
    random.shuffle(all_keys)  # 무작위 섞기

    N = len(all_keys)
    n_val = int(N * val_ratio)
    n_test = int(N * test_ratio)

    val_keys = all_keys[:n_val]
    test_keys = all_keys[n_val:n_val + n_test]
    train_keys = all_keys[n_val + n_test:]

    return train_keys, val_keys, test_keys

def merge_trials(trial_dict, keys):
    X_list, y_list = [], []
    for k in keys:
        X_list.append(trial_dict[k]['X'])
        y_list.append(trial_dict[k]['y'])
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def plot_window(X_window, y_window=None, title='One Window', sampling_rate=100):
    """
    X_window: shape (50, 8)
    y_window: shape (50,) or scalar
    """
    time = np.arange(X_window.shape[0]) / sampling_rate

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Acc
    axes[0].plot(time, X_window[:, 0], label='Acc X')
    axes[0].plot(time, X_window[:, 1], label='Acc Y')
    axes[0].plot(time, X_window[:, 2], label='Acc Z')
    axes[0].set_ylabel('Accel [m/s²]')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Thigh Acceleration')

    # Gyro
    axes[1].plot(time, X_window[:, 3], label='Gyro X')
    axes[1].plot(time, X_window[:, 4], label='Gyro Y')
    axes[1].plot(time, X_window[:, 5], label='Gyro Z')
    axes[1].set_ylabel('Gyro [rad/s]')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Thigh Angular Velocity')

    # Theta
    axes[2].plot(time, X_window[:, 6], label='Theta Est', color='purple')
    axes[2].plot(time, X_window[:,7], label='Theta dot Est', color='purple')
    axes[2].set_ylabel('Angle [rad]')
    axes[2].legend(loc='upper right')
    axes[2].set_title('Estimated Thigh Angle')

    # Gait phase
    if y_window is not None:
        if np.ndim(y_window) == 1:
            axes[3].plot(time, y_window, label='Gait Phase', color='green')
        else:
            axes[3].hlines(y_window, time[0], time[-1], label='Gait Phase', color='green')  # scalar case
        axes[3].legend(loc='upper right')
    axes[3].set_ylabel('Gait Phase')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_title('Gait Phase Output')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_sda_losses(train_losses, val_losses, src_losses, tgt_losses, mmd_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(src_losses, label='Source Loss')
    plt.plot(tgt_losses, label='Target Loss')
    plt.plot(mmd_losses, label='mmd Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SDA Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_circular_rmse(y_true, y_pred):
    """
    0~1 범위의 주기적(circular) 값들에 대해 RMSE를 계산.
    예: gait phase, angle (normalized), etc.

    Parameters:
        y_true: np.ndarray of shape (N, ...) — ground truth (in [0, 1])
        y_pred: np.ndarray of same shape — prediction (in [0, 1])

    Returns:
        float — circular RMSE
    """
    diff = np.abs(y_true - y_pred)
    wrapped_diff = np.minimum(diff, 100.0 - diff)
    return np.sqrt(np.mean(wrapped_diff ** 2))


def gait_phase_cal(y):
    phase = np.mod(np.arctan2(y[:, 1], y[:, 0])*50/np.pi,100)
    return phase

# 학습 루틴
def train_two_stage_sda(
    X_src_train, X_tgt_train, y_tgt_train,
    # X_tgt_val, y_tgt_val,
    stage1_epochs=50, stage2_epochs=50,
    patience=5, batch_size=64, verbose=True,
    learning_rate=1e-3, dropout=0.3, latent_dim=32
):
    input_shape = X_src_train.shape[1:]
    encoder, latent_dim = build_encoder(input_shape, dropout=dropout, latent_dim=latent_dim)
    # encoder, latent_dim = build_cnn_encoder(input_shape, dropout=dropout, latent_dim=latent_dim)
    decoder_tgt = build_decoder(latent_dim)
    # decoder_tgt = build_lstm_decoder(latent_dim)

    optimizer_enc = tf.keras.optimizers.Adam(learning_rate)
    optimizer_dec = tf.keras.optimizers.Adam(learning_rate)

    train_losses, val_losses = [], []
    mmd_losses = []
    best_val_loss = np.inf
    wait = 0

    # Stage 1: Train encoder using MMD only
    for epoch in range(stage1_epochs):
        idx_src = np.random.permutation(len(X_src_train))
        idx_tgt = np.random.permutation(len(X_tgt_train))
        n_batches = min(len(idx_src), len(idx_tgt)) // batch_size

        epoch_mmd_loss = 0

        for i in range(n_batches):
            b_src = idx_src[i * batch_size:(i + 1) * batch_size]
            b_tgt = idx_tgt[i * batch_size:(i + 1) * batch_size]
            Xs, Xt = X_src_train[b_src], X_tgt_train[b_tgt]

            with tf.GradientTape() as tape:
                zs = encoder(Xs, training=True)
                zt = encoder(Xt, training=True)
                mmd_loss = compute_mmd_loss(zs, zt)

            grads = tape.gradient(mmd_loss, encoder.trainable_variables)
            optimizer_enc.apply_gradients(zip(grads, encoder.trainable_variables))
            epoch_mmd_loss += mmd_loss.numpy()

        avg_mmd = epoch_mmd_loss / n_batches
        mmd_losses.append(avg_mmd)

        if verbose:
            print(f"[Stage 1 | Epoch {epoch+1}] MMD Loss: {mmd_loss.numpy():.4f}")

    # Stage 2: Freeze encoder, train decoder on target prediction
    encoder.trainable = False
    best_val_loss = np.inf
    wait = 0

    for epoch in range(stage2_epochs):
        idx = np.random.permutation(len(X_tgt_train))
        n_batches = len(idx) // batch_size
        epoch_loss = 0

        for i in range(n_batches):
            b = idx[i * batch_size:(i + 1) * batch_size]
            Xb, yb = X_tgt_train[b], y_tgt_train[b]

            with tf.GradientTape() as tape:
                z = encoder(Xb, training=False)
                y_pred = decoder_tgt(z, training=True)
                loss = tf.reduce_mean(tf.keras.losses.mse(yb, y_pred))

            grads = tape.gradient(loss, decoder_tgt.trainable_variables)
            optimizer_dec.apply_gradients(zip(grads, decoder_tgt.trainable_variables))
            epoch_loss += loss.numpy()
        train_loss = epoch_loss / n_batches
        # Validation
        # z_val = encoder(X_tgt_val, training=False)
        # y_val_pred = decoder_tgt(z_val, training=False)
        # val_loss = np.mean((y_tgt_val - y_val_pred.numpy()) ** 2)
        train_losses.append(epoch_loss / n_batches)
        # val_losses.append(val_loss)

        if verbose:
            print(f"[Stage 2 | Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}")

        # Early stopping
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print("⏹ Early stopping triggered!")
                break

    return encoder, decoder_tgt, train_losses, val_losses, mmd_losses

# 시각화 함수
def visualize_latent_space(encoder, X_src, X_tgt):
    z_src = encoder(X_src, training=False).numpy()
    z_tgt = encoder(X_tgt, training=False).numpy()
    z = np.concatenate([z_src, z_tgt], axis=0)
    labels = ['Source'] * len(z_src) + ['Target'] * len(z_tgt)

    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)

    plt.figure(figsize=(6, 6))
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(z_pca[idxs, 0], z_pca[idxs, 1], label=label, alpha=0.6)
    plt.title("Latent Space (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def send_email(sender= "aronwos1212@gmail.com", receiver= "aronwos1212@gmail.com", subject = "[Notification] NN Training completed",
               body="All trials completed.", app_password="tepx bjdq lgik xpdj"):

    # SMTP 서버 설정 (Gmail 예시)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # MIMEText 객체 생성 (메일 내용)
    msg = MIMEText(body, "plain", "utf-8")
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    # SMTP 서버 연결 및 메일 전송
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # TLS 보안 설정
        server.login(sender, app_password)
        server.sendmail(sender, receiver, msg.as_string())
        print("메일 전송 완료")
    except Exception as e:
        print(f"메일 전송 실패: {e}")
    finally:
        server.quit()

# 결과 저장 함수
def save_results_for_subject(subj_id, encoder, decoder_src, decoder_tgt,
                              train_losses, val_losses,
                              src_losses, tgt_losses, mmd_losses,
                              X_tgt_train, y_tgt_train,
                              X_tgt_val, y_tgt_val,
                              X_tgt_test, y_tgt_test,
                              base_path="./saved_models"):
    subj_path = os.path.join(base_path, subj_id)
    os.makedirs(subj_path, exist_ok=True)

    encoder.save(os.path.join(subj_path, "encoder.h5"))
    decoder_src.save(os.path.join(subj_path, "decoder_src.h5"))
    decoder_tgt.save(os.path.join(subj_path, "decoder_tgt.h5"))

    with open(os.path.join(subj_path, "results_data.pkl"), "wb") as f:
        pickle.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "src_losses": src_losses,
            "tgt_losses": tgt_losses,
            "mmd_losses": mmd_losses,
            "X_tgt_train": X_tgt_train,
            "y_tgt_train": y_tgt_train,
            "X_tgt_val": X_tgt_val,
            "y_tgt_val": y_tgt_val,
            "X_tgt_test": X_tgt_test,
            "y_tgt_test": y_tgt_test
        }, f)

# 전체 Grid Search 실행 함수
def run_and_save_all_combinations(subject_ids, lambda_mmd_list, lambda_src_list, lambda_tgt_list,
                                  stroke_dataset, X_src_train, y_src_train,
                                  train_fn, save_path="./saved_models", combine=False, encoder_units=[128, 64], decoder_units=[32], num_epoch=50):
    results = []
    for subj_id in subject_ids:
        print(f"📦 Running for Subject {subj_id}...")
        X_tgt_train, y_tgt_train = stroke_dataset[subj_id]['train']
        X_tgt_val, y_tgt_val = stroke_dataset[subj_id]['val']
        X_tgt_test, y_tgt_test = stroke_dataset[subj_id]['test']

        # If combine is True, merge source and target training data
        if combine:
            X_tgt_train = np.concatenate([X_src_train, X_tgt_train], axis=0)
            y_tgt_train = np.concatenate([y_src_train, y_tgt_train], axis=0)

        X_tgt_train, _, X_tgt_test, scaler_tgt = normalize_dataset(X_tgt_train, None, X_tgt_test)
        X_src_train, _, _, scaler_src = normalize_dataset(X_src_train)

        for lambda_mmd, lambda_src, lambda_tgt in product(lambda_mmd_list, lambda_src_list, lambda_tgt_list):
            encoder, decoder_src, decoder_tgt, train_losses, val_losses, src_losses, tgt_losses, mmd_losses = train_fn(
                X_src_train, y_src_train,
                X_tgt_train, y_tgt_train,
                # X_tgt_val, y_tgt_val,
                lambda_tgt=lambda_tgt,
                lambda_mmd=lambda_mmd,
                lambda_src=lambda_src,
                epochs=num_epoch, batch_size=64, verbose=False, learning_rate=0.001, patience=5, encoder_units=encoder_units, decoder_units=decoder_units
            )

            _, mse_tgt = evaluate_on_testset(encoder, decoder_tgt, X_tgt_test, y_tgt_test, name=f"{subj_id} Test")

            save_results_for_subject(
                f"{subj_id}_mmd{lambda_mmd}_src{lambda_src}_tgt{lambda_tgt}",
                encoder, decoder_src, decoder_tgt,
                train_losses, val_losses,
                src_losses, tgt_losses, mmd_losses,
                X_tgt_train, y_tgt_train,
                X_tgt_val, y_tgt_val,
                X_tgt_test, y_tgt_test,
                base_path=save_path
            )

            results.append({
                "subject": subj_id,
                "lambda_mmd": lambda_mmd,
                "lambda_src": lambda_src,
                "lambda_tgt": lambda_tgt,
                "mse": mse_tgt
            })
        sender = "aronwos1212@gmail.com"
        receiver = "aronwos1212@gmail.com"
        subject = "[Notification] NN Training completed"
        password = "tepx bjdq lgik xpdj"
        body = f"Subject{subj_id} trials completed."
        send_email(sender=sender, receiver=receiver, subject=subject,body=body,app_password=password)

    return results

def load_stroke_h5_grouped_by_subject_and_trial_hip(file_path, window_size=50, stride=1, pfx = 'paretic'):
    """
    Load stroke dataset from H5 file and organize it as:
    {
        'S001': {
            'T001': {'X': ..., 'y': ..., 'gc': ...},
            'T002': ...
        },
        ...
    }
    """
    subject_dict = {}
    with h5py.File(file_path, 'r') as f:
        for trial_name in f:
            grp = f[trial_name]

            # Extract subject ID and trial ID
            parts = trial_name.split('_')
            subject_id = parts[0]  # e.g., S001
            trial_id = parts[1]    # e.g., T001

            # Determine paretic side
            paretic_side = str(grp['paretic_side'][()].decode('utf-8')) \
                if isinstance(grp['paretic_side'][()], bytes) \
                else str(grp['paretic_side'][()])

            # Assign key prefix based on paretic side
            
            def get_data(prefix, key):
                return np.array(grp[f"{prefix}_{key}"]).squeeze()
            # Inputs
            X = np.stack([
                get_data(pfx, 'acc_x'),
                get_data(pfx, 'acc_y'),
                get_data(pfx, 'acc_z'),
                get_data(pfx, 'gyr_x'),
                get_data(pfx, 'gyr_y'),
                get_data(pfx, 'gyr_z'),
                get_data(pfx, 'hip_angle'),
                get_data(pfx, 'hip_angleV'),
            ], axis=1)

            # Gait phase
            phase_raw = np.array(grp['gc_hs']).squeeze()
            y = np.stack([np.cos(2 * np.pi * phase_raw), np.sin(2 * np.pi * phase_raw)], axis=1)

            # Windowing
            if len(X) < window_size:
                continue

            X_list, y_list, gc_list = [], [], []
            for start in range(0, len(X) - window_size + 1, stride):
                x_win = X[start:start + window_size]
                y_label = y[start + window_size - 1]
                gc_val = phase_raw[start + window_size - 1]
                X_list.append(x_win)
                y_list.append(y_label)
                gc_list.append(gc_val)

            if X_list:
                if subject_id not in subject_dict:
                    subject_dict[subject_id] = {}
                subject_dict[subject_id][trial_id] = {
                    'X': np.array(X_list),
                    'y': np.array(y_list),
                    'gc': np.array(gc_list)
                }

    return subject_dict

def load_h5_to_trial_dict_with_squeeze_hip(file_path, window_size=50, stride=5, max_walking_speed = 0.7, gc_type='gcR_hs'):
    trial_dict = {}
    with h5py.File(file_path, 'r') as f:
        for trial_name in f.keys():
            grp = f[trial_name]
            keys = list(grp.keys())

            # 보행 속도 확인
            if 'walking_speed' not in keys:
                continue
            walking_speed = np.mean(np.array(grp['walking_speed']).squeeze())
            if np.isscalar(walking_speed):
                speed_val = walking_speed
            else:
                speed_val = walking_speed[0]  # 혹시 리스트여도 첫 값 사용

            if speed_val > max_walking_speed:
                print(f"average walking speed is {walking_speed}")
                continue  # 보행 속도가 너무 빠르면 제외

            if all(k in keys for k in ['thigh_acc_x', 'thigh_acc_y', 'thigh_acc_z',
                                       'thigh_gyr_x', 'thigh_gyr_y', 'thigh_gyr_z',
                                       'theta_est', gc_type]):
                def get_data(key):
                    return np.array(grp[key]).squeeze()

                X = np.stack([
                    get_data('thigh_acc_x'),
                    get_data('thigh_acc_y'),
                    get_data('thigh_acc_z'),
                    get_data('thigh_gyr_x'),
                    get_data('thigh_gyr_y'),
                    get_data('thigh_gyr_z'),
                    get_data('hip_angle'),
                    get_data('hip_angleV'),
                ], axis=1)

                phase = get_data(gc_type) * 2 * np.pi * 0.01
                y = np.stack([np.cos(phase), np.sin(phase)], axis=1)

                if len(X) < window_size:
                    continue

                X_list, y_list = [], []
                for start in range(0, len(X) - window_size + 1, stride):
                    x_win = X[start:start + window_size]
                    y_label = y[start + window_size - 1]
                    X_list.append(x_win)
                    y_list.append(y_label)

                if X_list:
                    trial_dict[trial_name] = {
                        # 'X': np.array(X_list),
                        # 'y': np.array(y_list)
                        'X': np.stack(X_list).astype(np.float32),
                        'y': np.stack(y_list).astype(np.float32)
                    }
    return trial_dict