import numpy as np
from scipy.linalg import expm

def van_loan_discretization(dt, A, B=None, Q=None):
    n = A.shape[0]
    def get_input_mapping(A, B):
        if len(B.shape) == 1:
            B = B[np.newaxis].T
        m = B.shape[1]
        F_row_1 = np.hstack((A, B))
        F_row_2 = np.hstack((np.zeros((m,n)), np.zeros((m,m))))
        F = np.vstack((F_row_1, F_row_2))
        Fd = expm(F*dt)
        Ad = Fd[:n, :n]
        Bd = Fd[:n, n:]
        return Ad, Bd
    def get_noise_mapping(A, Q):
        F_row_1 = np.hstack((-A, Q))
        F_row_2 = np.hstack((np.zeros((n, n)), A.T))
        F = np.vstack((F_row_1, F_row_2))
        Fd = expm(F*dt)
        Ad = Fd[n:, n:].T
        Qd = Ad.dot(Fd[:n, n:])
        return Ad, Qd
    if B is not None:
        Ad, Bd = get_input_mapping(A, B)
    else:
        Bd = None
    if Q is not None:
        Ad, Qd = get_noise_mapping(A, Q)
    else:
        Qd = None
    return Ad, Bd, Qd

def sksym(x):
    return np.array([[0, -x[2], x[1]],[x[2], 0, -x[0]], [-x[1], x[0], 0]])

class Node(object):
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.children = None

    def add_child(self, new_child):
        if self.children is None:
            self.children = []
        self.children.append(new_child)

    def set_parent(self, new_parent):
        self.parent = new_parent
