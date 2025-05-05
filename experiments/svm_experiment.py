import datetime
import itertools
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random

import matplotlib.pyplot as plt
import mlflow
import mne
import numpy as np
import torch
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
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
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from utils.args_parser import get_args_parser
from data.dataloader import OddOneOutSignalDataLoader
from utils.visualization_utils import plot_pca_for_arrays, plot_pca_comparisons
from utils.utils import (
    generate_combinations,
    get_transport,
)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# To setup an mlflow server: mlflow server --host 127.0.0.1 --port 8080

seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

args = get_args_parser()


if __name__ == "__main__":
    data_dir = args.data_path
    label_file = os.path.join(data_dir, "labels.csv")
    dataset_name = args.dataset_name

    dataloader = OddOneOutSignalDataLoader(data_dir, label_file)
    subjects = [
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ]

    mlflow.set_experiment(
        f"SVM_{dataset_name}_{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
    )
    try:
        for i, subject in enumerate(subjects):
            with mlflow.start_run(run_name=f"Test Subject {subject}"):
                try:
                    args.use_EMD_Laplace = True

                    mlflow.log_params(params=args.__dict__)

                    (
                        train_set,
                        train_labels,
                        test_set,
                        test_labels,
                        train_subjects,
                        test_subjects,
                    ) = dataloader.get_folds(
                        test_subjects=(
                            int(subject)
                            if dataset_name in ["VEPESS"]
                            else subject
                        ),
                        standardize_by="subject",
                        encode_labels="label",
                        return_subjects=True,
                    )

                    X_trains = np.stack(train_set)
                    y_trains = train_labels
                    X_test = np.stack(test_set)
                    y_test = test_labels

                    # X_trains = X_trains[(y_trains == 0) + (y_trains == 1)]
                    # train_subjects = train_subjects[(y_trains == 0) + (y_trains == 1)]
                    # y_trains = y_trains[(y_trains == 0) + (y_trains == 1)]
                    #
                    # X_test = X_test[(y_test == 0) + (y_test == 1)]
                    # y_test = y_test[(y_test == 0) + (y_test == 1)]

                    if dataset_name == "VEPESS":
                        nfilter = [8]  # , 12, 16]
                        cov_est = ["scm"]  # , "lwf", "oas"]
                        c = [0.001, 0.01, 0.1, 1.0, 10, 100]
                        gammas = ["scale"]  # , "auto"]
                        kernel = ["linear"]
                        param_grid = {
                            "xdawncovariances__nfilter": nfilter,
                            "xdawncovariances__estimator": cov_est,
                            "svc__C": c,
                            "svc__gamma": gammas,
                            "svc__kernel": kernel,
                        }
                    elif dataset_name == "BCICa" or dataset_name == "BCICb":
                        nfilter = [4, 6, 8, 12, 16, 22]
                        c = [0.01, 0.1, 1.0, 10.0]
                        gammas = ["scale"]
                        kernel = ["linear"]
                        if len(np.unique(y_trains)) != 2:
                            param_grid = {
                                "csp__n_components": nfilter,
                                "onevsrestclassifier__estimator__C": c,
                                "onevsrestclassifier__estimator__gamma": gammas,
                                "onevsrestclassifier__estimator__kernel": kernel,
                            }
                        else:
                            param_grid = {
                                "csp__n_components": nfilter,
                                "svc__C": c,
                                "svc__gamma": gammas,
                                "svc__kernel": kernel,
                            }

                    # Classifier without OT to search parameters
                    combinations = generate_combinations(param_grid)
                    print("Number of combinations", len(combinations))

                    cv = itertools.islice(
                        LeaveOneGroupOut().split(X_trains, y_trains, train_subjects),
                        args.cv_folds,
                    )
                    if dataset_name == "VEPESS":
                        estimator = make_pipeline(
                            XdawnCovariances(),
                            TangentSpace(metric="riemann"),
                            SVC(
                                class_weight="balanced",
                                probability=False,
                            ),
                        )
                    elif dataset_name == "BCICa" or dataset_name == "BCICb":
                        estimator = make_pipeline(
                            mne.decoding.CSP(
                                reg="empirical",
                                log=True,
                                norm_trace=False,
                                cov_est="epoch",
                            ),
                            (
                                OneVsRestClassifier(
                                    SVC(
                                        class_weight="balanced",
                                        probability=True,
                                        max_iter=10000,
                                        verbose=False,
                                    )
                                )
                                if len(np.unique(y_trains)) != 2
                                else SVC(
                                    class_weight="balanced",
                                    probability=False,
                                    max_iter=10000,
                                )
                            ),
                        )


                    model = GridSearchCV(
                        estimator,
                        param_grid,
                        scoring="roc_auc" if dataset_name == "VEPESS" else "accuracy",
                        n_jobs=-1,
                        verbose=0,
                        cv=cv,
                        refit=True,
                    )

                    model.fit(X_trains, y_trains)
                    best_params = model.best_params_
                    print("Best score for noOT", model.best_score_)
                    print("Best params for noOT", best_params)

                    y_pred = model.predict(X_test)

                    # Validate best model
                    cm = ConfusionMatrixDisplay.from_predictions(
                        y_test, y_pred, normalize="true"
                    )
                    mlflow.log_figure(cm.figure_, "noOT_cm.png")
                    if dataset_name == "VEPESS":
                        roc = RocCurveDisplay.from_predictions(
                            y_test, model.decision_function(X_test), pos_label=1
                        )
                        mlflow.log_figure(roc.figure_, "noOT_roc.png")

                    y_pred = model.predict(X_test)
                    noOT_metrics = {
                        "noOT BA": balanced_accuracy_score(y_test, y_pred),
                        "noOT accuracy": accuracy_score(y_test, y_pred),
                        "noOT precision": precision_score(
                            y_test,
                            y_pred,
                            average=(
                                "macro" if len(np.unique(y_trains)) != 2 else "binary"
                            ),
                        ),  # VEPESS esetén nem kell average
                        "noOT recall": recall_score(
                            y_test,
                            y_pred,
                            average=(
                                "macro" if len(np.unique(y_trains)) != 2 else "binary"
                            ),
                        ),  # VEPESS esetén nem kell average
                        "noOT AUC": roc_auc_score(
                            y_test,
                            (
                                model.predict_proba(X_test)
                                if len(np.unique(y_trains)) != 2
                                else model.decision_function(X_test)
                            ),
                            average="macro",
                            multi_class=(
                                "ovr" if len(np.unique(y_trains)) != 2 else "raise"
                            ),
                        ),
                        "noOT f1-score": fbeta_score(
                            y_test,
                            y_pred,
                            beta=1,
                            average=(
                                "macro" if len(np.unique(y_trains)) != 2 else "binary"
                            ),
                        ),
                    }
                    mlflow.log_metrics(noOT_metrics)
                    mlflow.log_dict(
                        {"noOT best parameters": best_params}, "noOT_best_params.json"
                    )
                    print("NO OT METRICS:", noOT_metrics)
                except Exception as e:
                    print("ERROR IN NO OT")
                    print(e)
                    raise e

                ## OT #################################################################################

                try:
                    if dataset_name == "VEPESS":
                        laplace_reg = [1e-3, 1e-2]
                        bary_reg = [5e-2]
                    else:
                        laplace_reg = [0.1, 1.0, 10]
                        bary_reg = [0.2]
                    laplace_similarity = [10, 20, 30]
                    laplace_reg_type = ["disp"]
                    laplace_alpha = [0.5]
                    ot_params = {
                        "Laplace_similarity_param": laplace_similarity,
                        "Laplace_reg": laplace_reg,
                        "Laplace_alpha": laplace_alpha,
                        "Laplace_reg_type": laplace_reg_type,
                        "bary_reg": bary_reg,
                    }

                    classifier_param_grid = {
                        k: v
                        for k, v in param_grid.items()
                        if "classifier" in k or "svc" in k
                    }
                    print(classifier_param_grid)
                    ot_param_combinations = generate_combinations(ot_params)

                    if dataset_name == "VEPESS":
                        feature_extractor = make_pipeline(
                            XdawnCovariances(
                                **{
                                    k.split("__")[-1]: v
                                    for k, v in best_params.items()
                                    if "xdawncovariances" in k
                                }
                            ),
                            TangentSpace(metric="riemann"),
                        )
                    elif dataset_name == "BCICa" or dataset_name == "BCICb":
                        feature_extractor = make_pipeline(
                            mne.decoding.CSP(
                                **{
                                    k.split("__")[-1]: v
                                    for k, v in best_params.items()
                                    if "csp" in k
                                }
                            ),
                        )
                    else:
                        feature_extractor = make_pipeline(
                            FunctionTransformer(np.mean, kw_args={"axis": 1}),
                            StandardScaler(),
                        )

                    Xs_np = feature_extractor.fit_transform(X_trains, y_trains)
                    Xs = [
                        Xs_np[train_subjects == subject]
                        for subject in np.unique(train_subjects)
                    ]
                    ys = [
                        y_trains[train_subjects == subject]
                        for subject in np.unique(train_subjects)
                    ]
                    Xt = feature_extractor.transform(X_test)
                    best_ot_params = None
                    best_score = 0.0
                    best_classifier_params = None
                    for ot_param_combination in ot_param_combinations:
                        for key, param in ot_param_combination.items():
                            args.__setattr__(key, param)

                        transport = get_transport(ys, args)

                        transport.fit(Xs=Xs, ys=ys, Xt=Xt)

                        cv = itertools.islice(
                            LeaveOneGroupOut().split(
                                X_trains, y_trains, train_subjects
                            ),
                            args.cv_folds,
                        )
                        estimator = (
                            OneVsRestClassifier(
                                SVC(
                                    class_weight="balanced",
                                    probability=(
                                        True if len(np.unique(y_trains)) != 2 else False
                                    ),
                                    max_iter=10000,
                                )
                            )
                            if len(np.unique(y_trains)) != 2
                            else SVC(
                                class_weight="balanced",
                                probability=False,
                                verbose=False,
                            )
                        )
                        classifier = GridSearchCV(
                            make_pipeline(estimator),
                            classifier_param_grid,
                            n_jobs=-1,
                            cv=cv,
                            scoring=(
                                "roc_auc" if dataset_name == "VEPESS" else "accuracy"
                            ),
                            verbose=0,
                            refit=False,
                        )
                        classifier.fit(transport.transform(Xs=Xs), y_trains)
                        if classifier.best_score_ >= best_score:
                            best_score = classifier.best_score_
                            best_ot_params = ot_param_combination
                            best_classifier_params = classifier.best_params_

                    for key, param in best_ot_params.items():
                        args.__setattr__(key, param)
                    transport = get_transport(ys, args)
                    transport.fit(Xs=Xs, ys=ys, Xt=Xt)

                    estimator = (
                        OneVsRestClassifier(
                            SVC(
                                class_weight="balanced",
                                probability=(
                                    True if len(np.unique(y_trains)) != 2 else False
                                ),
                                max_iter=10000,
                            )
                        )
                        if len(np.unique(y_trains)) != 2
                        else SVC(
                            class_weight="balanced", probability=False
                        )
                    )
                    model = make_pipeline(estimator)
                    model.set_params(**best_classifier_params)
                    transported_source = transport.transform(Xs=Xs)
                    model.fit(transported_source, y_trains)
                    y_pred = model.predict(Xt)

                    cm_ot = ConfusionMatrixDisplay.from_predictions(
                        y_test, y_pred, normalize="true"
                    )
                    mlflow.log_figure(cm_ot.figure_, "OT_cm.png")
                    if dataset_name == "VEPESS":
                        roc = RocCurveDisplay.from_predictions(
                            y_test, model.decision_function(Xt), pos_label=1
                        )
                        mlflow.log_figure(roc.figure_, "OT_roc.png")

                    bary_plot = plot_pca_for_arrays(
                        arrays=[transport.Xbar, transported_source, Xt],
                        n_components=2,
                        labels=[y_trains, y_trains, y_test],
                    )
                    mlflow.log_figure(bary_plot, "PCA.png")
                    plot_pca_comparisons(X1=np.concatenate(Xs),
                                         y1=y_trains,
                                         X2=transport.Xbar,
                                         y2=y_trains,
                                         X3=transported_source,
                                         y3=y_trains,
                                         X4=Xt,
                                         y4=y_test,
                                         subject=subject,
                                         use_mlflow=True)
                    metrics = {
                        "OT BA": balanced_accuracy_score(y_test, y_pred),
                        "OT accuracy": accuracy_score(y_test, y_pred),
                        "OT precision": precision_score(
                            y_test,
                            y_pred,
                            average=(
                                "macro" if len(np.unique(y_trains)) != 2 else "binary"
                            ),
                        ),
                        "OT recall": recall_score(
                            y_test,
                            y_pred,
                            average=(
                                "macro" if len(np.unique(y_trains)) != 2 else "binary"
                            ),
                        ),
                        "OT AUC": roc_auc_score(
                            y_test,
                            (
                                # model.decision_function(Xt)
                                model.predict_proba(Xt)
                                if len(np.unique(y_trains)) != 2
                                else model.decision_function(Xt)
                            ),
                            multi_class=(
                                "ovr" if len(np.unique(y_trains)) != 2 else "raise"
                            ),
                        ),
                        "OT f1-score": fbeta_score(
                            y_test,
                            y_pred,
                            beta=1,
                            average=(
                                "macro" if len(np.unique(y_trains)) != 2 else "binary"
                            ),
                        ),
                    }
                    mlflow.log_metrics(metrics)
                    best_params.update(best_ot_params)
                    mlflow.log_dict(
                        {"OT best parameters": best_params}, "OT_best_params.json"
                    )
                    print("OT METRICS:", metrics)
                    plt.close("all")
                except Exception as e:
                    print("ERROR IN OT")
                    print(str(e))
                    raise e

    finally:
        pass
