# Stabilizer-Ground-states

This is the GitHub repo for "Stabilizer ground states: theory, algorithms and applications" by Jiace Sun, Lixue Cheng, and Shi-xin Zhang. [Paper Link](https://arxiv.org/abs/2403.08441)

## Usage

### python (for demonstration)
```
cd py
python sparse.py path_to_lib/example/hamiltonian_finite.txt
python klocal_sparse.py path_to_lib/example/hamiltonian_finite.txt
python klocal_periodic.py path_to_lib/example/hamiltonian_periodic.txt
```
### C++ (for performance)
#### Installation
requirement: Eigen3 >= 3.4.0, G++ >= 9, Boost, OpenMP (for parallelism)
```
cd cpp
mkdir build
cd build
cmake ..
make
```
#### Run
```
./klocal_sparse path_to_lib/example/hamiltonian_finite.txt
./sparse path_to_lib/example/hamiltonian_finite.txt
```

### Format of Hamiltonian
```
n k
w1 P1 sites1
w2 P2 sites2
...
```
n = number of qubits for finite Hamiltonians, or period for periodic Hamiltonians

k = locality of Hamiltonian (i.e. each Pauli must be in site m to m + k - 1 for some m)

See details in example
