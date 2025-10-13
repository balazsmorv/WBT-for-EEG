#!/usr/bin/env python3
# nn_experiment.py
import datetime
import itertools
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlflow
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    fbeta_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from utils.args_parser import get_args_parser
from data.dataloader import OddOneOutSignalDataLoader
from utils.visualization_utils import plot_pca_for_arrays3, plot_pca_comparisons
from util_fns import generate_combinations, get_transport

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# ------------------------------
# Reproducibility
# ------------------------------
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

args = get_args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Models
# ------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 128
    hidden: int = 128
    dropout: float = 0.2
    patience: int = 10


# ------------------------------
# Utilities
# ------------------------------
def build_feature_extractor(dataset_name: str, best_params: Dict = None):
    """
    Return an sklearn Pipeline that maps raw X to tabular feature vectors.
    Mirrors choices in SVM/LDA/LogReg so NN is comparable.
    """
    if dataset_name == "VEPESS":
        xdawn_kwargs = {}
        if best_params is not None:
            xdawn_kwargs = {k.split("__")[-1]: v for k, v in best_params.items() if "xdawncovariances" in k}
        return make_pipeline(
            XdawnCovariances(**xdawn_kwargs),
            TangentSpace(metric="riemann"),
        )
    elif dataset_name in ["BCICa", "BCICb"]:
        csp_kwargs = {}
        if best_params is not None:
            csp_kwargs = {k.split("__")[-1]: v for k, v in best_params.items() if "csp" in k}
        return make_pipeline(
            mne.decoding.CSP(
                reg="empirical",
                log=True,
                norm_trace=False,
                cov_est="epoch",
                **csp_kwargs,
            )
        )
    elif dataset_name == "SEED":
        return make_pipeline(
            FunctionTransformer(np.mean, kw_args={"axis": 1}),
            StandardScaler(),
        )
    else:
        # Fallback: flatten epochs (sensible default)
        return FunctionTransformer(lambda X: X.reshape(X.shape[0], -1))


def _to_tensor(np_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np_array.astype(np.float32))


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: TrainConfig,
) -> Tuple[float, Dict]:
    in_dim = X_train.shape[1]
    model = MLP(in_dim, num_classes, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xtr, ytr = _to_tensor(X_train).to(device), torch.from_numpy(y_train).long().to(device)
    Xva, yva = _to_tensor(X_val).to(device), torch.from_numpy(y_val).long().to(device)

    best_val = -np.inf
    best_state = None
    epochs_no_improve = 0

    for _ in range(cfg.epochs):
        model.train()
        idx = torch.randperm(Xtr.size(0), device=device)
        for start in range(0, Xtr.size(0), cfg.batch_size):
            batch_idx = idx[start : start + cfg.batch_size]
            xb = Xtr[batch_idx]
            yb = ytr[batch_idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation each epoch
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == yva).float().mean().item()

        # early stopping
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val, {"state_dict": model.state_dict(), "in_dim": in_dim, "num_classes": num_classes, "cfg": cfg.__dict__}


def select_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
    dataset_name: str,
) -> TrainConfig:
    """Light subject-wise CV over a tiny grid to keep runtime in check."""
    if dataset_name == "VEPESS":
        grid = {
            "lr": [1e-3],
            "weight_decay": [1e-4],
            "hidden": [256],
            "dropout": [0.2],
            "batch_size": [64],
            "epochs": [150],
        }
    elif dataset_name in ["BCICa", "BCICb"]:
        grid = {
            "lr": [1e-3],
            "weight_decay": [1e-4],
            "hidden": [256],
            "dropout": [0.2],
            "batch_size": [64],
            "epochs": [150],
        }
    else:  # SEED or others
        grid = {
            "lr": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-5],
            "hidden": [128],
            "dropout": [0.2, 0.5],
            "batch_size": [128],
            "epochs": [100],
        }

    # Build combos
    keys, values = zip(*grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -np.inf
    best_cfg = None

    # Use one subject-wise fold per combo (args.cv_folds controls which split)
    for combo in combos:
        cfg = TrainConfig(**combo)
        logo = LeaveOneGroupOut()
        cv = itertools.islice(logo.split(X, y, groups), 1)  # one fold per combo
        try:
            tr_idx, va_idx = next(cv)
        except StopIteration:
            # degenerate fallback
            n = len(X)
            perm = np.random.permutation(n)
            cut = max(1, int(0.8 * n))
            tr_idx, va_idx = perm[:cut], perm[cut:]

        val_acc, _ = train_one_fold(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], num_classes, cfg)
        if val_acc > best_score:
            best_score = val_acc
            best_cfg = cfg

    return best_cfg if best_cfg is not None else TrainConfig()


def fit_final_model(X_train: np.ndarray, y_train: np.ndarray, cfg: TrainConfig, num_classes: int):
    # 10% holdout from training for early stopping
    n = X_train.shape[0]
    n_val = max(1, int(0.1 * n))
    idx = np.random.permutation(n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    best_val, artifacts = train_one_fold(
        X_train[tr_idx], y_train[tr_idx],
        X_train[val_idx], y_train[val_idx],
        num_classes, cfg
    )

    model = MLP(artifacts["in_dim"], num_classes, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    model.load_state_dict(artifacts["state_dict"])
    model.eval()
    return model


def predict_logits(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(_to_tensor(X).to(device))
        return logits.detach().cpu().numpy()


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    data_dir = args.data_path
    label_file = os.path.join(data_dir, "labels.csv")
    dataset_name = args.dataset_name

    dataloader = OddOneOutSignalDataLoader(data_dir, label_file)
    subjects = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    mlflow.set_experiment(f"NN_{dataset_name}_{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}")

    for i, subject in enumerate(subjects):
        with mlflow.start_run(run_name=f"Test Subject {subject}"):
            try:
                mlflow.log_params(params=args.__dict__)

                (
                    train_set,
                    train_labels,
                    test_set,
                    test_labels,
                    train_subjects,
                    test_subjects,
                ) = dataloader.get_folds(
                    test_subjects=(int(subject) if dataset_name in ["VEPESS", "SEED"] else subject),
                    standardize_by="subject",
                    encode_labels="label",
                    return_subjects=True,
                )

                X_tr_raw = np.stack(train_set)
                y_tr = np.asarray(train_labels)
                X_te_raw = np.stack(test_set)
                y_te = np.asarray(test_labels)

                # X_tr_raw = X_tr_raw[(y_tr == 0) + (y_tr == 1)]
                # train_subjects = train_subjects[(y_tr == 0) + (y_tr == 1)]
                # y_tr = y_tr[(y_tr == 0) + (y_tr == 1)]
                # X_te_raw = X_te_raw[(y_te == 0) + (y_te == 1)]
                # y_te = y_te[(y_te == 0) + (y_te == 1)]

                # ---- Base (no OT) pipeline
                feature_extractor = build_feature_extractor(dataset_name)
                X_tr = feature_extractor.fit_transform(X_tr_raw, y_tr)
                X_te = feature_extractor.transform(X_te_raw)

                num_classes = int(len(np.unique(y_tr)))

                # Hyperparam selection
                cfg = select_hyperparams(X_tr, y_tr, train_subjects, num_classes, dataset_name)
                mlflow.log_params(
                    {
                        "nn_lr": cfg.lr,
                        "nn_weight_decay": cfg.weight_decay,
                        "nn_epochs": cfg.epochs,
                        "nn_batch_size": cfg.batch_size,
                        "nn_hidden": cfg.hidden,
                        "nn_dropout": cfg.dropout,
                        "nn_patience": cfg.patience,
                    }
                )

                # Train & eval
                model = fit_final_model(X_tr, y_tr, cfg, num_classes)
                logits = predict_logits(model, X_te)
                y_pred = logits.argmax(axis=1)

                noOT_metrics = {
                    "noOT BA": balanced_accuracy_score(y_te, y_pred),
                    "noOT accuracy": accuracy_score(y_te, y_pred),
                    "noOT precision": precision_score(
                        y_te, y_pred, average=("macro" if num_classes != 2 else "binary")
                    ),
                    "noOT recall": recall_score(
                        y_te, y_pred, average=("macro" if num_classes != 2 else "binary")
                    ),
                    "noOT AUC": roc_auc_score(
                        y_te,
                        (F.softmax(torch.from_numpy(logits), dim=1).numpy() if num_classes != 2 else logits[:, 1]),
                        average=("macro" if num_classes != 2 else "macro"),
                        multi_class=("ovr" if num_classes != 2 else "raise"),
                    ),
                    "noOT f1-score": fbeta_score(
                        y_te, y_pred, beta=1, average=("macro" if num_classes != 2 else "binary")
                    ),
                }
                mlflow.log_metrics(noOT_metrics)

                cm = ConfusionMatrixDisplay.from_predictions(y_te, y_pred, normalize="true")
                mlflow.log_figure(cm.figure_, "noOT_cm.png")
                if dataset_name == "VEPESS" and num_classes == 2:
                    roc = RocCurveDisplay.from_predictions(y_te, logits[:, 1], pos_label=1)
                    mlflow.log_figure(roc.figure_, "noOT_roc.png")

                # ---- Optional OT (WBT) variant
                try:
                    if dataset_name == "VEPESS":
                        laplace_reg = [1e-3, 1e-2]
                        bary_reg = [5e-2]
                    else:
                        laplace_reg = [0.1, 1.0]
                        bary_reg = [0.2]
                    laplace_similarity = [10, 20]
                    laplace_reg_type = ["disp"]
                    laplace_alpha = [0.5]
                    ot_params = {
                        "Laplace_similarity_param": laplace_similarity,
                        "Laplace_reg": laplace_reg,
                        "Laplace_alpha": laplace_alpha,
                        "Laplace_reg_type": laplace_reg_type,
                        "bary_reg": bary_reg,
                    }

                    ot_param_combos = generate_combinations(ot_params)

                    # Rebuild features fresh for OT path
                    feature_extractor_ot = build_feature_extractor(dataset_name)
                    Xs_np = feature_extractor_ot.fit_transform(X_tr_raw, y_tr)
                    Xs = [Xs_np[train_subjects == s] for s in np.unique(train_subjects)]
                    ys_list = [y_tr[train_subjects == s] for s in np.unique(train_subjects)]
                    Xt = feature_extractor_ot.transform(X_te_raw)

                    best_ot_score = -np.inf
                    best_ot_params = None

                    # Probe model for quick selection
                    probe_cfg = TrainConfig(
                        lr=max(cfg.lr, 5e-4),
                        weight_decay=cfg.weight_decay,
                        epochs=max(60, cfg.epochs // 2),
                        batch_size=cfg.batch_size,
                        hidden=max(64, cfg.hidden // 2),
                        dropout=cfg.dropout,
                        patience=cfg.patience,
                    )

                    for combo in ot_param_combos:
                        for k, v in combo.items():
                            setattr(args, k, v)
                        transport = get_transport(ys_list, args)
                        transport.fit(Xs=Xs, ys=ys_list, Xt=Xt)

                        Xs_bar = transport.transform(Xs=Xs)
                        if isinstance(Xs_bar, list):
                            Xs_bar = np.concatenate(Xs_bar, axis=0)

                        logo = LeaveOneGroupOut()
                        inner_cv = itertools.islice(logo.split(X_tr, y_tr, train_subjects), 1)
                        try:
                            tr_idx, va_idx = next(inner_cv)
                        except StopIteration:
                            n = len(X_tr)
                            perm = np.random.permutation(n)
                            cut = max(1, int(0.8 * n))
                            tr_idx, va_idx = perm[:cut], perm[cut:]

                        val_acc, _ = train_one_fold(Xs_bar[tr_idx], y_tr[tr_idx], Xs_bar[va_idx], y_tr[va_idx], num_classes, probe_cfg)
                        if val_acc > best_ot_score:
                            best_ot_score = val_acc
                            best_ot_params = combo

                    if best_ot_params is not None:
                        for k, v in best_ot_params.items():
                            setattr(args, k, v)
                        transport = get_transport(ys_list, args)
                        transport.fit(Xs=Xs, ys=ys_list, Xt=Xt)

                        transported_source = transport.transform(Xs=Xs)
                        if isinstance(transported_source, list):
                            transported_source = np.concatenate(transported_source, axis=0)

                        model_ot = fit_final_model(transported_source, y_tr, cfg, num_classes)
                        logits_ot = predict_logits(model_ot, Xt)
                        y_pred_ot = logits_ot.argmax(axis=1)

                        metrics = {
                            "OT BA": balanced_accuracy_score(y_te, y_pred_ot),
                            "OT accuracy": accuracy_score(y_te, y_pred_ot),
                            "OT precision": precision_score(
                                y_te, y_pred_ot, average=("macro" if num_classes != 2 else "binary")
                            ),
                            "OT recall": recall_score(
                                y_te, y_pred_ot, average=("macro" if num_classes != 2 else "binary")
                            ),
                            "OT AUC": roc_auc_score(
                                y_te,
                                (F.softmax(torch.from_numpy(logits_ot), dim=1).numpy() if num_classes != 2 else logits_ot[:, 1]),
                                average=("macro" if num_classes != 2 else "macro"),
                                multi_class=("ovr" if num_classes != 2 else "raise"),
                            ),
                            "OT f1-score": fbeta_score(
                                y_te, y_pred_ot, beta=1, average=("macro" if num_classes != 2 else "binary")
                            ),
                        }
                        mlflow.log_metrics(metrics)

                        cm_ot = ConfusionMatrixDisplay.from_predictions(y_te, y_pred_ot, normalize="true")
                        mlflow.log_figure(cm_ot.figure_, "OT_cm.png")

                        bary_plot = plot_pca_for_arrays3(
                            arrays=[transport.Xbar, transported_source, Xt],
                            n_components=2,
                            labels=[y_tr, y_tr, y_te],
                        )
                        mlflow.log_figure(bary_plot, "PCA.png")
                        plot_pca_comparisons(
                            X1=feature_extractor.fit_transform(X_tr_raw, y_tr),
                            y1=y_tr,
                            X2=transport.Xbar,
                            y2=y_tr,
                            X3=transported_source,
                            y3=y_tr,
                            X4=Xt,
                            y4=y_te,
                            subject=subject,
                            use_mlflow=True,
                        )

                        mlflow.log_dict({"OT best parameters": best_ot_params}, "OT_best_params.json")

                except Exception as e:
                    print("ERROR in OT pipeline:", str(e))  # non-fatal

                print(f"Completed NN run for subject {subject}")

            except Exception as e:
                print("ERROR in NN run:", str(e))
                raise e