import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="""Path to folder containing the data files""",
    )
    parser.add_argument(
        "--dataset_name",
        default="BCIC",
        type=str,
        help="""Name of the dataset""",
        choices=["VEPESS", "BCICa", "BCICb", "SEED"],
    )
    parser.add_argument(
        "--bary_solver_numitermax",
        default=10e10,
        type=float,
        help="""Maximum number of iterations for the barycenter solver""",
    )
    parser.add_argument(
        "--bary_reg",
        default=10.0,
        type=float,
        help="""Reguarization parameter for the barycenter computation""",
    )
    parser.add_argument(
        "--bary_stop_thr",
        default=1e-10,
        type=float,
        help="""Stop error threshold for the barycenter computation""",
    )
    parser.add_argument(
        "--EMD_maxiter",
        default=1e15,
        type=float,
        help="""Max iter number for the EMD transport""",
    )
    parser.add_argument("--verbose", default=True, type=bool, help="""Verbose""")
    parser.add_argument(
        "--use_bary_labels",
        default=True,
        type=bool,
        help="""Whether to use the train labels to compute a better separated barycenter""",
    )
    parser.add_argument(
        "--use_EMD_Laplace",
        default=True,
        type=bool,
        help="""Whether to use Laplace regularized OT instead of EMD""",
    )
    parser.add_argument(
        "--Laplace_reg",
        default=1.0,
        type=float,
        help="""Class regularization for EMD Laplace transform""",
    )
    parser.add_argument(
        "--Laplace_reg_src",
        default=0.001,
        type=float,
        help="""Source relative importance in Laplace regularization""",
    )
    parser.add_argument(
        "--Laplace_reg_type",
        default="pos",
        type=str,
        choices=["pos", "disp"],
        help="""Class regularization term""",
    )
    parser.add_argument(
        "--Laplace_similarity_param",
        default=5,
        type=int,
        help="""Similarity parameter for Laplace regularization""",
    )
    parser.add_argument(
        "--Laplace_alpha",
        default=0.5,
        type=float,
        help="""Alpha parameter for Laplace regularization""",
    )
    parser.add_argument(
        "--cv_folds",
        default=3,
        type=int,
        help="""Number of folds in cross validation""",
    )
    return parser.parse_args()
