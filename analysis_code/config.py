"""Constants: the default (x_ss, x_os) and the literature spin-component scalings."""

from __future__ import annotations


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
