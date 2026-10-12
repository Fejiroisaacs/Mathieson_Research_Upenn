import argparse
import json
import os

import time

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers, regularizers
from keras.utils import pad_sequences, to_categorical

NUM_CLASSES = 5
MAXLEN = 5000 
L2 = 1e-4
KSIZE = 2
CLASS_NAMES = ["neutral", "hard", "hard-near", "soft", "soft-near"]


def pad_x(x):
    # each element is (n_sites, 208) int8; pad sites axis with 0s at the end.
    # int8 keeps a 3,000-example chunk at ~3 GB instead of ~12 GB (original used int32).
    return pad_sequences(x, padding="post", maxlen=MAXLEN, dtype="int8")


def pad_pos(p):
    return pad_sequences(p, padding="post", value=-1.0, dtype="float32", maxlen=MAXLEN)


def build_model(n_chrom):
    l2 = regularizers.l2(L2)

    # Branch 1: genotype matrix (sites x chromosomes), 1D convs along sites
    x_in = keras.Input(shape=(MAXLEN, n_chrom), name="genotypes")
    h = layers.Conv1D(256, KSIZE, activation="relu", kernel_regularizer=l2)(x_in)
    h = layers.Conv1D(256, KSIZE, activation="relu", kernel_regularizer=l2)(h)
    h = layers.MaxPooling1D(KSIZE)(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Conv1D(256, KSIZE, activation="relu", kernel_regularizer=l2)(h)
    h = layers.MaxPooling1D(KSIZE)(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Conv1D(256, KSIZE, activation="relu", kernel_regularizer=l2)(h)
    h = layers.AveragePooling1D(KSIZE)(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Conv1D(256, KSIZE, activation="relu", kernel_regularizer=l2)(h)
    h = layers.AveragePooling1D(KSIZE)(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Flatten()(h)

    # Branch 2: positions of segregating sites
    p_in = keras.Input(shape=(MAXLEN,), name="positions")
    p = layers.Dense(64, activation="relu", kernel_regularizer=l2)(p_in)
    p = layers.Dropout(0.1)(p)

    # Merge (was keras.layers.Merge(mode='concat'))
    m = layers.Concatenate()([h, p])
    m = layers.Dense(256, activation="relu", kernel_initializer="random_normal",
                     kernel_regularizer=l2)(m)
    m = layers.Dropout(0.25)(m)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(m)

    model = keras.Model([x_in, p_in], out, name="sweep_cnn")
    model.compile(loss="categorical_crossentropy",
                  # Keras 2.0.6 Adam defaults: lr=0.001, betas 0.9/0.999, epsilon=1e-8
                  # (Keras 3 default epsilon is 1e-7, so set it explicitly)
                  optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8),
                  metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to final.split.up.seln.big.npz")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="debug: only use first N training chunks (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keras.utils.set_random_seed(args.seed)

    # npz was written by Python 2 with object (ragged) arrays -> pickle + latin1
    u = np.load(args.data, encoding="latin1", allow_pickle=True)
    keys = list(u.keys())
    print("npz keys:", len(keys), sorted(keys)[:10], "...", flush=True)

    # validation set: first 2,000 test examples (same as original)
    xval = pad_x(u["xtest"][:2000])
    yval = to_categorical(u["ytest"][:2000], NUM_CLASSES)
    pval = pad_pos(u["postest"][:2000])
    n_chrom = xval.shape[2]
    print("val shapes", xval.shape, pval.shape, yval.shape, flush=True)

    # plain string sort, exactly like the original (xtrain_0, xtrain_102000, xtrain_105000, ...)
    xk = sorted([k for k in keys if "xtrain" in k])
    yk = sorted([k for k in keys if "ytrain" in k])
    pk = sorted([k for k in keys if "postrain" in k])
    assert len(xk) == len(yk) == len(pk), (len(xk), len(yk), len(pk))
    chunks = list(zip(xk, yk, pk))
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    print(f"{len(chunks)} training chunks", flush=True)

    model = build_model(n_chrom)
    model.summary()

    history = []
    t0 = time.time()
    for e in range(args.epochs):
        for i, (kx, ky, kp) in enumerate(chunks):
            xtr = pad_x(u[kx])
            ytr = to_categorical(u[ky], NUM_CLASSES)
            ptr = pad_pos(u[kp])
            print(f"epoch {e} chunk {i+1}/{len(chunks)} {kx} n={len(xtr)} "
                  f"elapsed={(time.time()-t0)/60:.1f} min", flush=True)
            h = model.fit([xtr, ptr], ytr, batch_size=args.batch_size, epochs=1,
                          verbose=2, validation_data=([xval, pval], yval))
            history.append({"epoch": e, "chunk": kx,
                            **{k: float(v[-1]) for k, v in h.history.items()}})
            del xtr, ytr, ptr
        model.save(os.path.join(args.out, f"sweep_cnn_epoch{e}.keras"))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    del xval, pval, yval

    # Final test: balanced 600 per class (3,000) from xfinal, same selection rule as original
    yfinal_all = np.asarray(u["yfinal"]).astype(int)
    counts, keep = [0] * NUM_CLASSES, []
    for idx, c in enumerate(yfinal_all):
        if counts[c] < 600:
            keep.append(idx)
            counts[c] += 1
        if sum(counts) >= 3000:
            break
    keep = np.array(keep)
    xf = pad_x(u["xfinal"][keep])
    pf = pad_pos(u["posfinal"][keep])
    yf = yfinal_all[keep]

    pred = np.argmax(model.predict([xf, pf], batch_size=256, verbose=0), axis=1)
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    cm = confusion_matrix(yf, pred, labels=list(range(NUM_CLASSES)))
    acc = accuracy_score(yf, pred)
    print("Final test confusion matrix (rows=true, cols=pred):", CLASS_NAMES)
    print(cm)
    print(f"Final test accuracy: {acc:.4f}  (paper: 0.606)")
    np.save(os.path.join(args.out, "final_confusion_matrix.npy"), cm)
    np.save(os.path.join(args.out, "final_test_pred.npy"), np.stack([yf, pred]))
    model.save(os.path.join(args.out, "sweep_cnn_final.keras"))
    with open(os.path.join(args.out, "final_metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "confusion_matrix": cm.tolist(),
                   "classes": CLASS_NAMES}, f, indent=1)


if __name__ == "__main__":
    main()