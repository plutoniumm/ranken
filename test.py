from ranken.core import State, normalise, dagger, GramSchmidt, Projector, Qdit
from ranken.utils import Loss, rand, minima
import numpy as np

N = 10
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

subspace_basis = np.array([PSI(i) for i in range(2)])
# subspace_basis = GramSchmidt(subspace_basis)

proj, proj_perp = Projector(basis=subspace_basis)

size = (2*D + 1)*(r - 1)

def phi_ik(X):
  lmda = X[0]
  qubit = Qdit(2, X[1:5].reshape(2, 2))
  qutrit = Qdit(3, X[5:11].reshape(3, 2))

  return lmda, qubit, qutrit

def f(X):
  L, qbit, qtrit = phi_ik(X)
  PHI_rx = normalise(L * np.kron(qbit, qtrit))

  return Loss(PHI_rx, proj_perp)

res = minima(f, rand(R_MAX, size))
minim = res.fun
params = res.x
print(f'E_r(θ={THETA:.2f}) = {minim:.4f} | Iters: {res.nfev}')
print(f"Hilbert Space sz: {size} @ 1Qubit + 1Qutrit")