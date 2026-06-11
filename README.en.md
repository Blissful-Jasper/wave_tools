# Wave Tools

[中文](README.md) | English

Wave Tools is a Python source package for tropical atmospheric wave diagnostics. It provides utilities for Wheeler-Kiladis wavenumber-frequency spectra, equatorial wave filtering, cross-spectral analysis, Matsuno dispersion curves, EOF analysis, phase composites, and common scientific plotting workflows.

The repository is currently maintained as a source tree and is intended for use from analysis scripts, notebooks, and model post-processing workflows.

## Features

- Wheeler-Kiladis spectral analysis with symmetric/antisymmetric decomposition, background smoothing, and plotting support
- Frequency-wavenumber filtering for Kelvin, ER, MRG, IG, TD, and MJO signals
- Cross spectrum, coherence, and phase diagnostics between two variables
- Matsuno shallow-water modes and dispersion curves
- EOF/PCA analysis, Kelvin-wave phase composites, and lag composites
- Maps, Hovmoller diagrams, WK spectra, Taylor diagrams, and vector legends
- Helper routines for converting ICON/HEALPix unstructured grids to regular latitude-longitude grids

## Installation

```bash
git clone git@github.com:Blissful-Jasper/wave_tools.git
python -m pip install -r wave_tools/requirements.txt
```

When running Python from the parent directory of the repository, the package can be imported directly:

```python
import wave_tools

print(wave_tools.get_version())
```

## Quick Examples

### Wheeler-Kiladis Spectrum

```python
import xarray as xr
from wave_tools.spectral import calculate_wk_spectrum

olr = xr.open_dataset("olr.nc")["olr"]

power_sym, power_asym, background = calculate_wk_spectrum(
    olr,
    window_days=96,
    skip_days=30,
    output_path="wk_spectrum.nc",
)
```

### Kelvin-Wave Filtering

```python
import xarray as xr
from wave_tools.filters import CCKWFilter

pr = xr.open_dataarray("pr_daily.nc")

filt = CCKWFilter(
    ds=pr,
    sel_dict={"lat": slice(-15, 15)},
    wave_name="kelvin",
    units="mm/day",
    spd=1,
    n_workers=4,
    verbose=True,
)

kelvin = filt.process()
kelvin.to_netcdf("pr_kelvin.nc")
```

### Cross-Spectral Analysis

```python
import xarray as xr
from wave_tools.cross_spectrum import quick_cross_spectrum

pr = xr.open_dataarray("pr.nc")
olr = xr.open_dataarray("olr.nc")

result = quick_cross_spectrum(
    pr,
    olr,
    segLen=96,
    segOverLap=-65,
    symmetry="symm",
)

coherence_sq = result["STC"].sel(component="COH2")
phase = result["STC"].sel(component="PHAS")
```

### Plot a WK Spectrum

```python
from wave_tools.plotting import plot_wk_spectrum

fig, axes = plot_wk_spectrum(
    power_sym,
    power_asym,
    background,
    wavenumber=power_sym["wavenumber"],
    frequency=power_sym["frequency"],
    add_matsuno_lines=True,
    save_path="wk_spectrum.png",
)
```

## Module Guide

| Module | Purpose |
| --- | --- |
| `spectral.py` | Wheeler-Kiladis spectral analysis |
| `filters.py` | Equatorial wave filtering and the CCKW filtering workflow |
| `cross_spectrum.py` | Cross spectrum, coherence, and phase for a pair of variables |
| `cross_spectrum_analysis.py` | Multi-experiment cross-spectral workflows and plotting helpers |
| `matsuno.py` | Matsuno theoretical modes and dispersion curves |
| `phase.py` | Peak detection, Kelvin-wave phase analysis, and composites |
| `eof.py` | EOF/PCA analysis |
| `plotting.py` | WK spectra, maps, Taylor diagrams, and related plotting functions |
| `Xianpumap.py` | Western Pacific basemaps and Hovmoller plotting helpers |
| `utils.py` | Data loading, model filtering, Radon diagnostics, and grid conversion |
| `diagnostics.py` | GMS and thermodynamic diagnostics |

## Data Conventions

Most routines expect an `xarray.DataArray` with dimension names that include:

- `time`
- `lat`
- `lon`

Filtering and spectral routines generally assume regular temporal sampling. In `CCKWFilter`, `spd` is the number of samples per day; use `spd=1` for daily data and `spd=4` for 6-hourly data.

## Dependencies

Core dependencies include `numpy`, `xarray`, `scipy`, `matplotlib`, `pandas`, `cartopy`, `numba`, `joblib`, `scikit-image`, and `healpy`. Some diagnostic routines also use `metpy` and `geocat-comp`.

## Acknowledgements

Parts of Wave Tools were integrated, adapted, and modified from the following public codebases. We thank the original authors for making their work available; copyrights and license terms remain with the respective upstream projects.

- [mmaiergerber/wk_spectra](https://github.com/mmaiergerber/wk_spectra)
- brianpm: [wavenumber_frequency_functions.py](https://github.com/Blissful-Jasper/wavenumber_frequency/blob/master/wavenumber_frequency_functions.py)
- Alejandro Jaramillo: [wk_analysis.py](https://github.com/Blissful-Jasper/wk_spectra/blob/master/wk_spectra/wk_analysis.py)
- muting-chien: [CCKW_aquaplanet/function](https://github.com/muting-chien/CCKW_aquaplanet/tree/ef3a5ea0f1166a831aa69b6834db3f52e0aa0c19/function)
- tmiyachi:[mcclimate](https://github.com/tmiyachi/mcclimate)

## Citation

If this package is used in a paper or report, please cite the relevant method papers as appropriate, for example:

- Wheeler, M. and Kiladis, G. N. (1999). Convectively coupled equatorial waves: Analysis of clouds and temperature in the wavenumber-frequency domain.
- Matsuno, T. (1966). Quasi-geostrophic motions in the equatorial area.

## Maintainer

Jianpu
Hohai University
xianpuji@hhu.edu.cn
