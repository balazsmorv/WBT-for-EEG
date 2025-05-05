from functools import partial
from itertools import product
import numpy as np
import ot

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
