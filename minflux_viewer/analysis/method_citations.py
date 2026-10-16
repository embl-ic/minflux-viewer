'Method citations'

from __future__ import annotations

import math

CITE_STDDEV = (
    "Ostersehlt et al., Nat. Methods 19:1072 (2022)",
    "https://doi.org/10.1038/s41592-022-01577-1",
)


CITE_DRIFT_TIME_WINDOWS = CITE_STDDEV


CITE_MINFLUX_STABILISATION = (
    "Schmidt et al., Nat. Commun. 12:1478 (2021)",
    "https://doi.org/10.1038/s41467-021-21652-z",
)


CITE_CRLB = (
    "Balzarotti et al., Science 355:606 (2017)",
    "https://doi.org/10.1126/science.aak9913",
)


CITE_CRLB_MARIN = (
    "Marin & Ries, arXiv:2410.12427 (2024)",
    "https://arxiv.org/abs/2410.12427",
)


CITE_SIMUFLUX = (
    "Marin & Ries, Nat. Commun. 17:246 (2026)",
    "https://doi.org/10.1038/s41467-025-66952-w",
)


_SIMUFLUX_DONUT_SIGMA_Q_NM = 310.0 / math.sqrt(8.0 * math.log(2.0))


CITE_FRC_BANTERLE = (
    "Banterle et al., J. Struct. Biol. 183:363 (2013)",
    "https://doi.org/10.1016/j.jsb.2013.05.004",
)


CITE_FRC_NIEUWENHUIZEN = (
    "Nieuwenhuizen et al., Nat. Methods 10:557 (2013)",
    "https://doi.org/10.1038/nmeth.2448",
)


ANISOTROPY_NOTE = (
    "Z scaling factor estimation from trace anisotropy (custom, MINFLUX Data "
    "Viewer): the factor is estimated from single-molecule traces by Gaussian fits to "
    "log-distance histograms of each localization's offset from its trace centroid "
    "(lateral vs. axial extent). It is applied as a Z-scaling view and never baked "
    "into raw coordinates. This empirical method does not calculate the factor "
    "from refractive indices."
)


NPC_NOTE = (
    "NPC ring-convolution segmentation (custom, MINFLUX Data Viewer): NPC centres are "
    "detected by convolving the 2-D localization histogram with a normalized donut "
    "kernel exp(-|x^2 + y^2 - (d/2)^2| / (4*rim^2)) that peaks at the ring radius, "
    "followed by local-maximum peak finding and a ring 'support score' (annulus angular "
    "coverage and radial fit) acceptance filter."
)


CITE_ANISOTROPY = (ANISOTROPY_NOTE, None)


CITE_NPC = (NPC_NOTE, None)
