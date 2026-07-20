# SubsetDIC

SubsetDIC is a Python package for 2D subset-based digital image correlation
(DIC). The performance-critical algorithms are implemented in C++ and exposed
to Python through pybind11.

This project is based on the algorithm flow of Ncorr and keeps the original
Ncorr reference source under `reference_code/` for comparison.

For the pure Python implementation, see
[lbd-hfut/Subset-DIC](https://github.com/lbd-hfut/Subset-DIC).

## Features

- Subset-based 2D DIC for grayscale image pairs.
- C++ core with Python API.
- ROI-based region extraction.
- SIFT or manual seed selection.
- Region-growing DIC propagation.
- Optional displacement-gradient / strain-field calculation.
- Example `ring` and `star` cases under `case/`.

## Project Layout

```text
SubsetDIC/
|-- src/
|   |-- subsetdic/        # Python API and configuration helpers
|   `-- cpp/              # C++ DIC core and pybind11 bindings
|-- config/
|   `-- default.yaml      # Default DIC parameters
|-- case/
|   |-- ring/             # Example ring images and ROI
|   `-- star/             # Example star displacement images and ROI
|-- tests/
|   |-- test_dic.py       # Synthetic translation integration test
|   `-- run_cases.py      # Run bundled ring/star cases
`-- reference_code/       # Ncorr MATLAB/C++ reference implementation
```

## Installation

Requirements:

- Python 3.9+
- CMake 3.21+
- A C++17 compiler
- Visual Studio C++ Build Tools on Windows

Install in editable mode:

```powershell
python -m pip install -e .
```

This compiles the C++ extension module `_core`. In editable mode, Python source
files are loaded from this repository, while the compiled extension may be
installed into your Python environment's `site-packages`, for example:

```text
C:\Users\<user>\miniconda3\Lib\site-packages\subsetdic\_core.cp312-win_amd64.pyd
```

After changing files in `src/cpp/`, run the install command again to rebuild
the extension.

## Quick Start

```python
import numpy as np
from PIL import Image
from subsetdic import SubsetDIC

ref = np.array(Image.open("case/star/001.bmp")).astype(np.float64)
cur = np.array(Image.open("case/star/002.bmp")).astype(np.float64)
roi = (np.array(Image.open("case/star/003.bmp")) > 128).astype(np.uint8)

dic = SubsetDIC({
    "dic": {
        "radius": 5,
        "spacing": 1,
        "cutoff_diffnorm": 1e-6,
        "cutoff_iteration": 50,
        "subsettrunc": False,
    },
    "seeds": {
        "method": "manual",
        "n_seeds": 1,
        "manual_positions": [(512, 128)],
    },
    "strain": {
        "enabled": True,
        "radius": 15,
    },
    "border": {
        "bcoef": 5,
        "interp": 5,
        "extrap": 5,
    },
})

result = dic.run(ref, cur, roi)

print(result.success)
print(result.u.shape, result.v.shape)
print(result.valid.sum())
```

## API

The main entry point is:

```python
from subsetdic import SubsetDIC

dic = SubsetDIC(config)
result = dic.run(ref_img, def_img, roi_mask=None)
```

Inputs:

- `ref_img`: reference image, 2D array.
- `def_img`: deformed/current image, 2D array with the same shape.
- `roi_mask`: optional 2D mask. Non-zero pixels are included in DIC.

`DicResult` fields:

- `success`: whether DIC produced a result.
- `u`, `v`: displacement fields.
- `corrcoef`: correlation residual field.
- `valid`: valid point mask.
- `dudx`, `dudy`, `dvdx`, `dvdy`: displacement gradients, available when
  strain calculation is enabled.
- `points_computed`: number of propagated valid points.
- `regions`: ROI regions generated from the mask.

## Configuration

Default parameters live in `config/default.yaml`.

```yaml
dic:
  radius: 20
  spacing: 1
  cutoff_diffnorm: 1.0e-6
  cutoff_iteration: 50
  subsettrunc: true
  direct_seed_grid: false

seeds:
  method: sift
  n_seeds: 1

strain:
  enabled: true
  radius: 5

postprocess:
  enabled: true
  max_iterations: 8
  min_neighbors: 3
  corrcoef_threshold: 2.0

border:
  bcoef: 20
  interp: 20
  extrap: 20
```

Important parameters:

- `dic.radius`: subset radius in pixels.
- `dic.spacing`: grid spacing between calculated DIC points.
- `dic.cutoff_diffnorm`: IC-GN convergence threshold.
- `dic.cutoff_iteration`: maximum IC-GN iterations.
- `dic.subsettrunc`: whether to truncate subsets near ROI boundaries.
- `dic.direct_seed_grid`: solve each grid point with seed-style local search
  instead of region-growing propagation. This is slower but more robust for
  periodic or highly nonuniform displacement fields such as the star case.
- `seeds.method`: `sift` for automatic seeds or `manual` for fixed positions.
- `seeds.manual_positions`: list of `(x, y)` seed coordinates when using
  manual seeds.
- `strain.enabled`: whether to compute displacement gradients.
- `strain.radius`: radius used for gradient fitting.
- `postprocess.enabled`: interpolate bad displacement points from neighboring
  valid points before strain calculation.
- `postprocess.max_iterations`: maximum neighbor-growing fill iterations.
- `postprocess.min_neighbors`: minimum valid 8-neighbors required to fill a bad
  point.
- `postprocess.corrcoef_threshold`: points with larger correlation residual are
  treated as bad during interpolation.

You can also override selected DIC parameters when calling `run`:

```python
result = dic.run(ref, cur, roi, radius=15, spacing=3, compute_strain=False)
```

## Running Tests

Run the synthetic translation test:

```powershell
python tests\test_dic.py
```

Expected output includes:

```text
PASSED
All tests passed!
```

If `pytest` is installed, the same test can be run with:

```powershell
python -m pytest -q
```

## Running Example Cases

Run bundled cases:

```powershell
python tests\run_cases.py
```

This runs:

- `case/ring`
- `case/star`

Generated outputs are saved under each case's `result/` directory:

- `u.npy`
- `v.npy`
- `corrcoef.npy`
- `valid.npy`
- `dudx.npy`, `dudy.npy`, `dvdx.npy`, `dvdy.npy` when strain is enabled
- `overview.png`

The `result/` directories are ignored by Git because they are generated files.

## Development Notes

- Python files under `src/subsetdic/` are used directly in editable installs.
- C++ changes under `src/cpp/` require rebuilding with `python -m pip install -e .`.
- The extension module is named `subsetdic._core`.
- Arrays passed to the C++ core are stored in column-major/Fortran layout to
  match the Ncorr-style indexing.
- `reference_code/` contains the original Ncorr source used to check algorithm
  behavior and edge cases.

## License

See `LICENSE`.
