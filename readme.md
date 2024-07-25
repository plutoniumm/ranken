<img src="./assets/icon.svg" width="100" height="100" align="right">

### Ranken
Finding entanglement rank

[![PyPI version](https://badge.fury.io/py/ranken.svg)](https://pypi.org/project/ranken/)

## Usage
After `pip install ranken`, you can use the package as follows:

### Create a State
```py
from ranken.core import State

Rn = 3
basis = np.eye(Rn)

THETA = np.pi/2
l = 3
A = State.create(State.Ket_0, basis[i%Rn])
B = State.create(State.Ket_1, basis[(i+1)%Rn])

return State.combine([A, B], [np.cos(THETA/2), np.sin(THETA/2)])
```

### Make Projectors
```py
from ranken.core import Projector

subspace_basis = np.array([PSI(i) for i in range(2)])
# subspace_basis = GramSchmidt(subspace_basis)

proj, proj_perp = Projector(basis=subspace_basis)
```

### Find Rank
```py
from ranken.utils import Loss, rand, minima
from ranken.core import Qdit

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
print(f'E_r(θ={THETA:.2f}) = {res.fun:.4f}')
```

## Dev
```sh
python3 test.py
```

## License
MIT
