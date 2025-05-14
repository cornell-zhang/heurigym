# Run Baseline

## Install `tourbar2`

An alpha-release Python interface of `tourbar2` can be tested through pip on Linux and MacOS:

```
git clone https://github.com/toulbar2/toulbar2.git
cd tourbar2
python3 -m pip install pytoulbar2
```
The first line is only useful for Linux distributions that ship "old" versions of pip.

Commands for compiling the Python API on Linux/MacOS with cmake (Python module in `lib/*/pytb2.cpython*.so`):

```
pip3 install pybind11
mkdir build
cd build
cmake -DPYTB2=ON ..
make
```
Move the cpython library and the experimental `pytoulbar2.py` python class wrapper in the folder of the python script that does `import pytoulbar2`.

## Example

First, compile `fappeval.c` using `gcc -o fappeval fappeval.c`.

Second, move the `.in` file you want to run under this directory.

Then, launch the baseline using `python3 fapp.py your_file.in 3 | awk -f ./sol2fapp.awk - your_file`. The output format will be
```
New solution: 523 (0 backtracks, 0 nodes, depth 2, 0.382 seconds)
 X1=(47, -1) X2=(100, 1) X3=(1, 1) X4=(1, -1)


*****************************

RESULTS: (THETA = 4 )

        Number of unsatisfied mandatory constraints: 0
        Minimum relaxation level: k* = 3
        Number of violations per level: 
                2 1 1 0 0 0 0 0 0 0 0 
        Number of violations at level 2 (k*-1) = 1
        Total number of violations for levels i < k*-1 = 3

```
