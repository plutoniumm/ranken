from utils import States, normalise, dagger, GramSchmidt
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
basis = np.eye(3)

KET_0 = States.Ket_0
KET_1 = States.Ket_1

def PSI(i):
  l = len(basis)
  i = i%l
  j = (i+1)%l
  return a*np.kron(KET_0, basis[i]) + b*np.kron(KET_1, basis[j])
# end PSI

subspace_basis = np.array([PSI(i) for i in range(2)])
# subspace_basis = GramSchmidt(subspace_basis)
print(subspace_basis)
raise SystemExit

projector = np.array([
  np.outer(subspace_basis[i], dagger(subspace_basis[i]))
    for i in range(2)
])
proj_perp = np.eye(6) - sum(projector)


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


TEST = False
if TEST:
  e = np.exp(1)
  pi = np.pi
  X = [69, e, pi, pi, e, 1, 0, 0, 0.5, pi**0.5, 0]
  L, qbit, qtrit = phi_ik(X)
  print(L)
  print(qbit)
  print(qtrit)
  PHI_rx = normalise(L * np.kron(qbit, qtrit))
  l = loss(PHI_rx, proj_perp)
  # print(f'Loss: {l:.4f}')
  print(f'Loss: {l}')
else:
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