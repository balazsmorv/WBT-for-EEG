import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product, islice
from einops import rearrange
import numpy as np
import ot
from pyriemann.estimation import XdawnCovariances, Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS, LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from utils.wbt import WassersteinBarycenterTransport
from utils.wbt_barycenters import sinkhorn_barycenter


def generate_combinations(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()

    # Generate all combinations using product
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]

    return combinations


def get_transport(ys: list[np.ndarray], args):
    barycenter_solver = partial(
        sinkhorn_barycenter,
        ys=ys,
        ybar=np.concatenate(ys),
        numItermax=args.bary_solver_numitermax,
        reg=args.bary_reg,
        stopThr=args.bary_stop_thr,
        verbose=args.verbose,
    )
    if args.use_EMD_Laplace:

        from ot.da import EMDLaplaceTransport

        print("Using EMD Laplace with args: ", args)
        transport_solver = partial(
            # SinkhornLaplaceTransport,
            EMDLaplaceTransport,
            metric="sqeuclidean",
            max_iter=100,
            max_inner_iter=100000,
            tol=1e-2,
            inner_tol=1e-9,
            similarity="knn",
            verbose=args.verbose,
            reg_type=args.Laplace_reg_type,
            reg_src=args.Laplace_alpha,
            similarity_param=args.Laplace_similarity_param,
            reg_lap=args.Laplace_reg,
        )
    else:
        print("Using regular EMD transport")
        transport_solver = partial(ot.da.EMDTransport, max_iter=args.EMD_maxiter)

    return WassersteinBarycenterTransport(
        verbose=args.verbose,
        barycenter_solver=barycenter_solver,
        transport_solver=transport_solver,
        barycenter_initialization="random_cls",
    )


def get_noOT_estimator_with_SVM(dataset_name, params):
    if dataset_name == "VEPESS":
        return CovBasedClassifier(
            covariance_estimator=XdawnCovariances(
                nfilter=params["covariance_estimator__nfilter"],
                estimator=params["covariance_estimator__estimator"],
                xdawn_estimator=params["covariance_estimator__estimator"],
            ),
            projector=TangentSpace(metric="riemann"),
            classifier=SVC(
                C=params["classifier__C"],
                gamma=params["classifier__gamma"],
                kernel=params["classifier__kernel"],
                class_weight="balanced",
            ),
            calc_cov_subjectwise=False,
        )
    elif dataset_name == "BCIC":
        return CovBasedClassifier(
            covariance_estimator=Covariances(
                estimator=params["covariance_estimator__estimator"],
            ),
            projector=CSP(
                metric=params["projector__metric"],
                nfilter=params["projector__nfilter"],
            ),
            classifier=OneVsRestClassifier(
                SVC(
                    C=params["classifier__estimator__C"],
                    gamma=params["classifier__estimator__gamma"],
                    kernel=params["classifier__estimator__kernel"],
                    class_weight="balanced",
                )
            ),
        )
    else:
        raise NotImplementedError


def get_OT_estimator_with_SVM(dataset_name, params, transport):
    if dataset_name == "VEPESS":
        return CovBasedOTClassifier(
            covariance_estimator=XdawnCovariances(
                nfilter=params["covariance_estimator__nfilter"],
                estimator=params["covariance_estimator__estimator"],
                xdawn_estimator=params["covariance_estimator__estimator"],
            ),
            projector=TangentSpace(metric="riemann"),
            transport=transport,
            classifier=SVC(
                C=params["classifier__C"],
                gamma=params["classifier__gamma"],
                kernel=params["classifier__kernel"],
                class_weight="balanced",
            ),
            calc_cov_subjectwise=False,
        )
    elif dataset_name == "BCIC":
        return CovBasedOTClassifier(
            covariance_estimator=Covariances(
                estimator=params["covariance_estimator__estimator"],
            ),
            projector=CSP(
                metric=params["projector__metric"],
                nfilter=params["projector__nfilter"],
            ),
            transport=transport,
            classifier=OneVsRestClassifier(
                SVC(
                    C=params["classifier__estimator__C"],
                    gamma=params["classifier__estimator__gamma"],
                    kernel=params["classifier__estimator__kernel"],
                    class_weight="balanced",
                )
            ),
        )
    else:
        raise NotImplementedError


def get_noOT_estimator_with_LDA(dataset_name, params):
    if dataset_name == "VEPESS":
        return CovBasedClassifier(
            covariance_estimator=XdawnCovariances(
                nfilter=params["covariance_estimator__nfilter"],
                estimator=params["covariance_estimator__estimator"],
                xdawn_estimator=params["covariance_estimator__estimator"],
            ),
            projector=TangentSpace(metric="riemann"),
            classifier=LinearDiscriminantAnalysis(
                solver=params["classifier__solver"],
                shrinkage=params["classifier__shrinkage"],
                covariance_estimator=(
                    {"oas": OAS(), "lwf": LedoitWolf()}[
                        params["classifier__covariance_estimator"]
                    ]
                    if params["classifier__covariance_estimator"] is not None
                    else None
                ),
            ),
            calc_cov_subjectwise=False,
        )
    elif dataset_name == "BCIC":
        return CovBasedClassifier(
            covariance_estimator=Covariances(
                estimator=params["covariance_estimator__estimator"],
            ),
            projector=CSP(
                metric=params["projector__metric"],
                nfilter=params["projector__nfilter"],
            ),
            classifier=LinearDiscriminantAnalysis(
                solver=params["classifier__solver"],
                shrinkage=params["classifier__shrinkage"],
                covariance_estimator=(
                    {"oas": OAS(), "lwf": LedoitWolf()}[
                        params["classifier__covariance_estimator"]
                    ]
                    if params["classifier__covariance_estimator"] is not None
                    else None
                ),
            ),
        )
    else:
        raise NotImplementedError


def get_OT_estimator_with_LDA(dataset_name, params, transport):
    if dataset_name == "VEPESS":
        return CovBasedOTClassifier(
            covariance_estimator=XdawnCovariances(
                nfilter=params["covariance_estimator__nfilter"],
                estimator=params["covariance_estimator__estimator"],
                xdawn_estimator=params["covariance_estimator__estimator"],
            ),
            projector=TangentSpace(metric="riemann"),
            transport=transport,
            classifier=LinearDiscriminantAnalysis(
                solver=params["classifier__solver"],
                shrinkage=params["classifier__shrinkage"],
                covariance_estimator=(
                    {"oas": OAS(), "lwf": LedoitWolf()}[
                        params["classifier__covariance_estimator"]
                    ]
                    if params["classifier__covariance_estimator"] is not None
                    else None
                ),
            ),
            calc_cov_subjectwise=False,
        )
    elif dataset_name == "BCIC":
        return CovBasedOTClassifier(
            covariance_estimator=Covariances(
                estimator=params["covariance_estimator__estimator"],
            ),
            projector=CSP(
                metric=params["projector__metric"],
                nfilter=params["projector__nfilter"],
            ),
            transport=transport,
            classifier=LinearDiscriminantAnalysis(
                solver=params["classifier__solver"],
                shrinkage=params["classifier__shrinkage"],
                covariance_estimator=(
                    {"oas": OAS(), "lwf": LedoitWolf()}[
                        params["classifier__covariance_estimator"]
                    ]
                    if params["classifier__covariance_estimator"] is not None
                    else None
                ),
            ),
        )
    else:
        raise NotImplementedError


def get_noOT_estimator_with_LOGREG(dataset_name, params):
    if dataset_name == "VEPESS":
        return CovBasedClassifier(
            covariance_estimator=XdawnCovariances(
                nfilter=params["covariance_estimator__nfilter"],
                estimator=params["covariance_estimator__estimator"],
                xdawn_estimator=params["covariance_estimator__estimator"],
            ),
            projector=TangentSpace(metric="riemann"),
            classifier=LogisticRegression(
                C=params["classifier__C"],
                solver=params["classifier__solver"],
                penalty=params["classifier__penalty"],
                class_weight="balanced",
                random_state=42,
            ),
            calc_cov_subjectwise=False,
        )
    elif dataset_name == "BCIC":
        return CovBasedClassifier(
            covariance_estimator=Covariances(
                estimator=params["covariance_estimator__estimator"],
            ),
            projector=CSP(
                metric=params["projector__metric"],
                nfilter=params["projector__nfilter"],
            ),
            classifier=LogisticRegression(
                C=params["classifier__C"],
                solver=params["classifier__solver"],
                penalty=params["classifier__penalty"],
                class_weight="balanced",
                random_state=42,
                multi_class="multinomial",
            ),
        )
    else:
        raise NotImplementedError


def get_OT_estimator_with_LOGREG(dataset_name, params, transport):
    if dataset_name == "VEPESS":
        return CovBasedOTClassifier(
            covariance_estimator=XdawnCovariances(
                nfilter=params["covariance_estimator__nfilter"],
                estimator=params["covariance_estimator__estimator"],
                xdawn_estimator=params["covariance_estimator__estimator"],
            ),
            projector=TangentSpace(metric="riemann"),
            transport=transport,
            classifier=LogisticRegression(
                C=params["classifier__C"],
                solver=params["classifier__solver"],
                penalty=params["classifier__penalty"],
                class_weight="balanced",
                random_state=42,
            ),
            calc_cov_subjectwise=False,
        )
    elif dataset_name == "BCIC":
        return CovBasedOTClassifier(
            covariance_estimator=Covariances(
                estimator=params["covariance_estimator__estimator"],
            ),
            projector=CSP(
                metric=params["projector__metric"],
                nfilter=params["projector__nfilter"],
            ),
            transport=transport,
            classifier=LogisticRegression(
                C=params["classifier__C"],
                solver=params["classifier__solver"],
                penalty=params["classifier__penalty"],
                class_weight="balanced",
                random_state=42,
                multi_class="multinomial",
            ),
        )
    else:
        raise NotImplementedError


def evaluate_params(
    param_set,
    Xs,
    ys,
    dataset_name,
    args,
    cv,
    get_noOT_classifier_func,
    get_OT_classifier_func,
    get_tranport_func=None,
):
    print("Iteration parameters", param_set)
    cv_scores = []

    if get_tranport_func is not None:
        args.Laplace_similarity_param = param_set["Laplace_similarity_param"]
        args.Laplace_reg_src = param_set["Laplace_reg_src"]
        args.bary_reg = param_set["bary_reg"]

    for i, (train_indices, test_indices) in enumerate(
        islice(cv.split(Xs, ys), args.cv_folds)
    ):
        print("Fold", i + 1)
        test_idx = test_indices[0]

        Xs_cv_train = [Xs[train_index] for train_index in train_indices]
        ys_cv_train = [ys[train_index] for train_index in train_indices]

        if get_tranport_func is None:
            model = get_noOT_classifier_func(dataset_name, param_set)
        else:
            transport_ = get_tranport_func(ys_cv_train, args)
            model = get_OT_classifier_func(
                dataset_name=dataset_name,
                params=param_set,
                transport=transport_,
            )

        try:
            if get_tranport_func is None:
                model.fit(Xs_cv_train, ys_cv_train)
            else:
                model.fit(Xs_cv_train, ys_cv_train, Xs[test_idx])
        except Exception as e:
            print("Skippin iteration due to failure'")
            print(e)
            break

        if dataset_name == "VEPESS":
            score = roc_auc_score(
                ys[test_idx],
                model.decision_function(Xs[test_idx]),
            )
        elif dataset_name == "BCIC":
            score = accuracy_score(
                ys[test_idx],
                model.predict(Xs[test_idx]),
            )

        cv_scores.append(score)
        print(f"Score: {score}")

    if len(cv_scores) < args.cv_folds:
        cv_scores = [0.0]

    return np.mean(cv_scores), param_set


def worker_function(
    params,
    Xs,
    ys,
    dataset_name,
    args,
    cv,
    get_noOT_classifier_func,
    get_OT_classifier_func,
    get_tranport_func,
):
    """
    Worker function that handles a single parameter combination evaluation
    """
    try:
        return evaluate_params(
            params,
            Xs=Xs,
            ys=ys,
            dataset_name=dataset_name,
            args=args,
            cv=cv,
            get_noOT_classifier_func=get_noOT_classifier_func,
            get_OT_classifier_func=get_OT_classifier_func,
            get_tranport_func=get_tranport_func,
        )
    except Exception as e:
        print(f"Error in worker: {str(e)}")
        return 0.0, params


def parallel_parameter_search_legacy(
    combinations,
    timeout,
    Xs,
    ys,
    dataset_name,
    args,
    cv,
    get_noOT_classifier_func,
    get_OT_classifier_func,
    get_tranport_func,
):
    """
    Handles parallel execution of parameter combinations with timeout
    """
    results = []
    n_cores = 8  # Leave one core free

    # Create partial function with fixed arguments
    worker_partial = partial(
        worker_function,
        Xs=Xs,
        ys=ys,
        dataset_name=dataset_name,
        args=args,
        cv=cv,
        get_noOT_classifier_func=get_noOT_classifier_func,
        get_OT_classifier_func=get_OT_classifier_func,
        get_tranport_func=get_tranport_func,
    )

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Submit all tasks
        future_to_params = {
            executor.submit(worker_partial, params): params for params in combinations
        }

        # Collect results with timeout
        for future in future_to_params:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except TimeoutError:
                print(f"Task timed out for parameters: {future_to_params[future]}")
                results.append((0.0, future_to_params[future]))
            except Exception as e:
                print(f"Task failed with error: {str(e)}")
                results.append((0.0, future_to_params[future]))

    return results


from concurrent.futures import ThreadPoolExecutor, as_completed


def target(worker_partial, params, q):
    try:
        result = worker_partial(params)
        q.put(result)
    except Exception as e:
        q.put((0.0, params))  # Or you can pass the error message if desired

def run_task_with_timeout(worker_partial, params, timeout):
    """
    Runs a task in a separate process and terminates it if it exceeds the timeout.
    Returns the task result or a failure tuple.
    """
    from multiprocessing import Process, Queue

    # Use a Queue to capture the result from the worker process
    q = Queue()

    # Start the worker process
    p = Process(target=partial(target, worker_partial, params, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        # Timeout exceeded: kill the process and return failure result
        print(f"Task timed out for parameters: {params}, terminating process.")
        p.terminate()
        p.join()
        return (0.0, params)
    else:
        # Process finished in time: retrieve result if available
        if not q.empty():
            return q.get()
        else:
            return (0.0, params)

def parallel_parameter_search(
    combinations,
    timeout,
    Xs,
    ys,
    dataset_name,
    args,
    cv,
    get_noOT_classifier_func,
    get_OT_classifier_func,
    get_tranport_func,
):
    """
    Handles parallel execution of parameter combinations with a timeout.
    When a task times out, its process is killed and replaced so that subsequent
    parameter iterations can start.
    """
    results = []
    n_cores = 8  # Leave a couple of cores free

    # Create a partial function with fixed arguments
    worker_partial = partial(
        worker_function,
        Xs=Xs,
        ys=ys,
        dataset_name=dataset_name,
        args=args,
        cv=cv,
        get_noOT_classifier_func=get_noOT_classifier_func,
        get_OT_classifier_func=get_OT_classifier_func,
        get_tranport_func=get_tranport_func,
    )

    # Use a thread pool to parallelize the spawning of tasks
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        # Submit tasks using our wrapper that enforces timeout and kills hung processes
        futures = {
            executor.submit(run_task_with_timeout, worker_partial, params, timeout): params 
            for params in combinations
        }
        for future in as_completed(futures):
            params = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task failed with error: {str(e)} for parameters: {params}")
                results.append((0.0, params))
    return results


class CovBasedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, covariance_estimator, projector, classifier, calc_cov_subjectwise=True
    ):
        super().__init__()
        self.covariance_estimator = covariance_estimator
        self.projector = projector
        self.classifier = classifier
        self.calc_cov_subjectwise = calc_cov_subjectwise

    def fit(self, X: list[np.ndarray], y: list[np.ndarray]) -> "CovBasedClassifier":
        # Covariance estimation
        if self.calc_cov_subjectwise:
            Xs_covariances = np.concatenate(
                [
                    self.covariance_estimator.fit(X_subject, y_subject).transform(
                        X_subject
                    )
                    for X_subject, y_subject in zip(X, y)
                ]
            )
        else:
            Xs_covariances = self.covariance_estimator.fit_transform(
                np.concatenate(X), np.concatenate(y)
            )

        # Projection into a latent feature space
        Xs_projections = self.projector.fit_transform(Xs_covariances, np.concatenate(y))

        # Classification
        self.classifier.fit(Xs_projections, np.concatenate(y))

        return self

    def _feature_extraction(self, X: np.ndarray) -> np.ndarray:
        X_covariances = self.covariance_estimator.transform(X)
        X_projections = self.projector.transform(X_covariances)
        return X_projections

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_projections = self._feature_extraction(X)
        return self.classifier.predict(X_projections)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_projections = self._feature_extraction(X)
        return self.classifier.predict_proba(X_projections)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_projections = self._feature_extraction(X)
        return self.classifier.decision_function(X_projections)


class CovBasedOTClassifier(CovBasedClassifier):
    def __init__(
        self,
        covariance_estimator,
        projector,
        classifier,
        transport,
        calc_cov_subjectwise=True,
    ):
        super().__init__(
            covariance_estimator, projector, classifier, calc_cov_subjectwise
        )
        self.transport = transport

    def fit(
        self, X: list[np.ndarray], y: list[np.ndarray], Xt: np.ndarray
    ) -> "CovBasedOTClassifier":
        subjects = np.concatenate(
            [
                [subject_idx] * len(subject_data)
                for subject_idx, subject_data in enumerate(X)
            ]
        )

        # Covariance estimation
        if self.calc_cov_subjectwise:
            Xs_covariances = np.concatenate(
                [
                    self.covariance_estimator.fit(X_subject, y_subject).transform(
                        X_subject
                    )
                    for X_subject, y_subject in zip(X, y)
                ]
            )
        else:
            Xs_covariances = self.covariance_estimator.fit_transform(
                np.concatenate(X), np.concatenate(y)
            )

        # Projection into a latent feature space
        Xs_projections = self.projector.fit_transform(Xs_covariances, np.concatenate(y))

        Xs_projections_list = [
            Xs_projections[subject_idx == subjects]
            for subject_idx in np.unique(subjects)
        ]

        # Optimal transport
        self.transport.fit(
            Xs=Xs_projections_list,
            ys=y,
            Xt=self.projector.transform(self.covariance_estimator.transform(Xt)),
        )

        # Classification
        self.classifier.fit(
            self.transport.transform(Xs_projections_list), np.concatenate(y)
        )

        return self
