"""Constants: physics defaults, fit settings, reporting caps."""

from __future__ import annotations

from scipy.constants import Avogadro, calorie, physical_constants


# Hartree -> kcal/mol from the CODATA values scipy ships (1 kcal = 4184 J).
KCAL_PER_HA: float = physical_constants["Hartree energy"][0] * Avogadro / (1000.0 * calorie)

# Threads for reading W-matrix files (nine small files per system; I/O bound).
LOAD_WORKERS: int = 8

DEFAULT_FIT_DATASET: str = "TAE_W4-17nonMR"

# (x_ss, x_os) of the SSD-OPMP3 row; x_ss is set per reference with --x-ss
# (README §2).
X_FIXED_DEFAULT: tuple[float, float] = (0.5, 1.0)

# Literature spin-component scalings for the fixed-coefficient models.
#   SCS-MP2  Grimme, J. Chem. Phys. 118, 9095 (2003).
#   SOS-MP2  Jung, Lochan, Dutoi, Head-Gordon, J. Chem. Phys. 121, 9793 (2004).
#   SCS-MP3  Grimme, J. Comput. Chem. 24, 1529 (2003): SCS-MP2 second order
#            plus a quarter of the total third order.
SCS_MP2_SS: float = 1.0 / 3.0
SCS_MP2_OS: float = 6.0 / 5.0
SOS_MP2_OS: float = 1.3
SCS_MP3_E3: float = 1.0 / 4.0

# How many non-finite reactions a report row names before summarising.
MAX_EXCLUDED_NAMED: int = 10

# Fits: x, x_ss, x_os are restricted to PARAM_BOUNDS; a fit ending on an
# edge is flagged (FitResult.at_bound).
PARAM_BOUNDS:     tuple[float, float] = (-1.0, 2.0)
BOUND_ACTIVE_TOL: float               = 1.0e-4     # "on" an edge if within

GRID_STEP:   float = 0.25                          # 2-D grid seed spacing
BRENT_XATOL: float = 1.0e-6                        # 1-D bounded Brent
NM_OPTIONS:  dict[str, float | int] = {            # 2-D Nelder-Mead
    "xatol": 1.0e-5, "fatol": 1.0e-7, "maxiter": 1000, "maxfev": 1000,
}
