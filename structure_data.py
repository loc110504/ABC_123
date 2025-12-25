import pandas as pd
import numpy as np
import os

DATA_DIR = "data"

# =========================
# READ RAW CSVs
# =========================
phase1_df = pd.read_csv(os.path.join(DATA_DIR, "phase1_label.csv"))
p2_train_df = pd.read_csv(os.path.join(DATA_DIR, "phase2_train.csv"))
p2_eval_df  = pd.read_csv(os.path.join(DATA_DIR, "phase2_eval.csv"))
p2_test_df  = pd.read_csv(os.path.join(DATA_DIR, "phase2_test.csv"))

# =========================
# CLASS LIST (FROM PHASE1)
# =========================
classes = sorted(phase1_df["labels"].dropna().unique())
class_to_idx = {c: i for i, c in enumerate(classes)}

print("Classes:", classes)

# =========================
# CORE FUNCTION
# =========================
def to_onehot(df, prefix):
    onehot = np.zeros((len(df), len(classes)), dtype=int)

    if "labels" in df.columns:
        for i, lbl in enumerate(df["labels"]):
            if pd.isna(lbl):
                continue
            onehot[i, class_to_idx[lbl]] = 1

    out = pd.DataFrame(onehot, columns=classes)

    out.insert(
        0,
        "image",
        prefix + "/" + df["ID"].str.replace(".jpg", "", regex=False)
    )
    return out

# =========================
# BUILD TRAIN (PHASE1 + PHASE2/TRAIN)
# =========================
phase1_onehot = to_onehot(phase1_df, prefix="phase1")
p2_train_onehot = to_onehot(p2_train_df, prefix="phase2/train")

train_onehot = pd.concat(
    [phase1_onehot, p2_train_onehot],
    ignore_index=True
)

# =========================
# BUILD VAL / TEST
# =========================
val_onehot = to_onehot(p2_eval_df, prefix="phase2/eval")
test_onehot = to_onehot(p2_test_df, prefix="phase2/test")  # all-zero

# =========================
# SAVE
# =========================
train_onehot.to_csv(os.path.join(DATA_DIR, "train_onehot.csv"), index=False)
val_onehot.to_csv(os.path.join(DATA_DIR, "val_onehot.csv"), index=False)
test_onehot.to_csv(os.path.join(DATA_DIR, "test_onehot.csv"), index=False)

print("✅ train / val / test CSVs created successfully")
