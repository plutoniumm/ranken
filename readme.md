<img src="./assets/icon.svg" width="100" height="100" align="right">

### Ranken
Finding entanglement rank

[![PyPI version](https://badge.fury.io/py/ranken.svg)](https://pypi.org/project/ranken/)

## Usage
After `pip install ranken`, you can use the package as follows:

### Create a State
```py
from ranken import State

l = 3
A = State.create(State.Ket_0, basis[i%l])
B = State.create(State.Ket_1, basis[(i+1)%l])

return State.combine([A, B], [a, b])
```

## License
MIT