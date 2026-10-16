"""
Step 2 - Train the Tiny CNN, choose the confidence threshold, export int8 TFLite.

Usage:  python 2_train_model.py [epochs]        (default 40)
Needs:  dataset_logmel.npz from step 1
Output: model_int8.tflite, model_data.h, norm.json, confusion_matrix.png, results.txt

Split (ESC-50 folds, no leakage between clips of the same source recording):
    train = folds 1-3 (incl. noise-mixed copies) | validation = fold 4 | test = fold 5
"""
import sys, json
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
tf.keras.utils.set_random_seed(42)
d = np.load("dataset_logmel.npz")
X, y, fold, aug, CLASSES = d["X"], d["y"], d["fold"], d["aug"], list(d["classes"])

tr, va, te = fold <= 3, fold == 4, fold == 5
mean, std = float(X[tr].mean()), float(X[tr].std())      # normalisation from TRAIN only
json.dump(dict(mean=mean, std=std), open("norm.json", "w"))
nx = lambda a: ((a - mean) / std)[..., None].astype(np.float32)
Xtr, Xva, Xte = nx(X[tr]), nx(X[va]), nx(X[te])
ytr, yva, yte = y[tr], y[va], y[te]
cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(4), y=ytr)))

model = tf.keras.Sequential([
    tf.keras.layers.Input(Xtr.shape[1:]),
    *[l for f in (8, 16, 32) for l in (
        tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2))],
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation="softmax"),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=64, class_weight=cw,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)], verbose=2)

out = []
def log(s=""): print(s); out.append(s)

# ---- confidence threshold from VALIDATION set -------------------------------------------
pva = model.predict(Xva, verbose=0)
conf, pred = pva.max(1), pva.argmax(1)
log("\nValidation: confidence threshold vs accuracy on accepted windows")
chosen = 0.5
for th in np.arange(0.50, 0.96, 0.05):
    m = conf >= th
    acc = (pred[m] == yva[m]).mean() if m.any() else float("nan")
    log(f"  th={th:.2f}  accepted={m.mean()*100:5.1f}%  accuracy={acc*100:5.1f}%")
    if m.any() and acc >= 0.95 and chosen == 0.5: chosen = round(float(th), 2)
log(f"Chosen confidence threshold (first with >=95% accepted accuracy, else 0.50): {chosen}")

# ---- TEST set (float model) --------------------------------------------------------------
pte = model.predict(Xte, verbose=0).argmax(1)
log("\nTEST (fold 5) - float model\n" + classification_report(yte, pte, target_names=CLASSES, digits=3))
cm = confusion_matrix(yte, pte); log("Confusion matrix (rows=true, cols=pred):\n" + str(cm))

# ---- int8 quantisation -------------------------------------------------------------------
rep = lambda: (([Xtr[i:i + 1]] for i in np.random.default_rng(0).choice(len(Xtr), 300, replace=False)))
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
blob = conv.convert()
open("model_int8.tflite", "wb").write(blob)
with open("model_data.h", "w") as f:
    f.write("// int8 TFLite model for TensorFlow Lite Micro\nalignas(16) const unsigned char g_model[] = {\n")
    for i in range(0, len(blob), 12): f.write("  " + ", ".join(f"0x{b:02x}" for b in blob[i:i + 12]) + ",\n")
    f.write(f"}};\nconst unsigned int g_model_len = {len(blob)};\n")

it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
inp, outp = it.get_input_details()[0], it.get_output_details()[0]
s_in, z_in = inp["quantization"]
q = lambda a: np.clip(np.round(a / s_in + z_in), -128, 127).astype(np.int8)
qp = []
for i in range(len(Xte)):
    it.set_tensor(inp["index"], q(Xte[i:i + 1])); it.invoke(); qp.append(it.get_tensor(outp["index"])[0].argmax())
qp = np.array(qp)
log(f"\nTEST (fold 5) - int8 model accuracy: {(qp == yte).mean()*100:.1f}%   (float: {(pte == yte).mean()*100:.1f}%)")
log(f"Model file size: {len(blob)/1024:.1f} KB   Input quant scale/zero-point: {s_in:.6f} / {z_in}")
log(f"Chosen confidence threshold: {chosen}")
open("results.txt", "w").write("\n".join(out))

fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30, ha="right"); ax.set_yticklabels(CLASSES)
for i in range(4):
    for j in range(4): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test fold)")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=200)
print("Saved model_int8.tflite, model_data.h, norm.json, results.txt, confusion_matrix.png")
