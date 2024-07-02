import numpy as np


def corr_matrix_from_cov(var_covar: np.array) -> np.array:
    diag_inv = np.diag(1 / np.sqrt(np.diag(var_covar)))
    return diag_inv @ var_covar @ diag_inv
