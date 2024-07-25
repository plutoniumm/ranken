from utils import State, normalise, dagger, GramSchmidt, Projector
from scipy.optimize import minimize
import numpy as np

N = 10
tol = 1e-3
R_MAX = 2**7
THETA = 1.5
d = 3 # dim of QuDit 2
D = 5 # Qubit + Qutrit
r = 2 # level of entanglement
a = np.cos(THETA/2)
b = np.sin(THETA/2)

l = 3
basis = np.eye(l)

def PSI(i):
  A = State.create(State.Ket_0, basis[i%l])
  B = State.create(State.Ket_1, basis[(i+1)%l])

  return State.combine([A, B], [a, b])
# end PSI

subspace_basis = np.array([PSI(i) for i in range(2)])
# subspace_basis = GramSchmidt(subspace_basis)

proj, proj_perp = Projector(basis=subspace_basis)

size = (2*D + 1)*(r - 1)

def phi_ik(X):
  lmda = X[0]
  psi2_0 = X[1] + 1j*X[2]
  psi2_1 = X[3] + 1j*X[4]
  qubit = np.array([psi2_0, psi2_1])
  qubit = normalise(qubit)

  psi3_0 = X[5] + 1j*X[6]
  psi3_1 = X[7] + 1j*X[8]
  psi3_2 = X[9] + 1j*X[10]
  qutrit = np.array([psi3_0, psi3_1, psi3_2])
  qutrit = normalise(qutrit)

  return lmda, qubit, qutrit

# <phi_rx|proj_perp|phi_rx>
def loss(phi_rx, proj):
  left = dagger(phi_rx)
  right = phi_rx

  prod = np.dot(np.dot(left, proj), right)

  return np.real_if_close(prod, tol=1e-6)

def minimize_wrapper(X):
  L, qbit, qtrit = phi_ik(X)
  PHI_rx = normalise(L * np.kron(qbit, qtrit))

  l = loss(PHI_rx, proj_perp)
  return l

x0 = np.random.randint(-R_MAX, R_MAX, size=size)
res = minimize(
  minimize_wrapper,
  x0=x0,
  method='L-BFGS-B',
  tol=tol
)
minima = res.fun
params = res.x
print(f'E_r(θ={THETA:.2f}) = {minima:.4f} | Iters: {res.nfev}')
print(f"Hilbert Space sz: {size} @ 1Qubit + 1Qutrit")