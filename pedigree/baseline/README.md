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

First, move the `.pre` file you want to run under this directory.
Next, launch the baseline using `python mendel.py your_file.pre`. The output format will be different from the ones provided in `\solutions`, e.g.,
```
New solution: 1 (0 backtracks, 0 nodes, depth 2, 0.136 seconds)
 a1a3 a5a7 a3a5 a4a8 a1a7 a1a5 a1a3 a1a5 a3a7 a4a8 a3a5 a5a8 a1a3 a3a4 a5a8 a3a7 a1a1 a3a4 a5a7 a1a4 a1a4 a1a5 a3a5 a1a8 a4a7 a1a1 a4a4 a3a8 a5a5 a3a3 a5a7 a5a7 a3a8 a1a4 a1a3 a4a5 a6a7 a1a5 a4a5 a1a5 a3a8 a1a7 a5a8 a4a8 a3a4 a4a8 a7a8 a5a7 a4a8
COST: 1
```
This can be easily mapped back to the `.sol` format using domain label mapping.