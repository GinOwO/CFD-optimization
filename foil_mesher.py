"""
This is a slightly modified (didn't like the many syscalls) version of the program at
    https://github.com/curiosityFluids/curiosityFluidsAirfoilMesher/blob/master/curiosityFluidsAirfoilMesher.py

Original License:
-----------------------------------------------------------------------------------
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

 This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import math
from pathlib import Path

import numpy as np


def meshify(Coords: np.ndarray, run_path: Path):
    ChordLength = 1
    DomainHeight = 20  # Multiples Chord Length
    WakeLength = 20  # Multiples Chord Length
    firstLayerHeight = 0.001  # Multiples Chord Length
    growthRate = 1.05
    MaxCellSize = 0.5  # Multiples of chordlength

    # These Values can be played with to improve mesh quality
    BLHeight = 0.1  # Fraction of chord length - Height of BL block
    LeadingEdgeGrading = 2
    TrailingEdgeGrading = 0.8
    inletGradingFactor = 0.5
    TrailingBlockAngle = 5  # Degrees

    NBL = int(
        np.rint(
            math.log(1 - (BLHeight / firstLayerHeight * (1 - growthRate)))
            / math.log(growthRate)
        )
    )
    MaxLayerThickness = firstLayerHeight * growthRate**NBL
    BLGrading = MaxLayerThickness / firstLayerHeight
    LFF = DomainHeight / 2 - BLHeight
    NFF = int(
        np.rint(
            math.log(MaxCellSize / MaxLayerThickness)
            / math.log(1 - MaxLayerThickness / LFF + MaxCellSize / LFF)
        )
    )
    farfieldGrowthRate = (MaxCellSize / MaxLayerThickness) ** (1 / NFF)
    FFGrading = MaxCellSize / MaxLayerThickness
    NFFA = int(np.rint(3.14159 / 2 * DomainHeight / 2 / MaxCellSize))

    # Import airfoil, split top and bottom, and split into quadrants
    X = Coords[:, 0]  # X Coordinates
    Y = Coords[:, 1]  # Y Coordinates
    numpoint = X.size  # Number of points
    xTop = np.array([])  # Initialize Top half X Coordinates
    yTop = np.array([])  # Initialize Top half Y Coordinates
    xBottom = np.array([])  # Initialize Bottom half X Coordinates
    yBottom = np.array([])  # Initialize Bottom half Y Coordinates
    TopCount = 0
    BottomCount = 0
    for i in range(numpoint):
        if X[i] - X[i - 1] < 0 and i != 0 and i != numpoint - 1:
            xTop = np.append(xTop, X[i])
            yTop = np.append(yTop, Y[i])
            TopCount += 1
        elif X[i] - X[i - 1] > 0 and i != 0 and i != numpoint - 1:
            xBottom = np.append(xBottom, X[i])
            yBottom = np.append(yBottom, Y[i])
            BottomCount += 1
    # South West Quadrant of Airfoil
    SWx = xBottom[xBottom < 0.25]
    SWy = yBottom[xBottom < 0.25]
    LSW = 0
    for i in range(len(SWx) - 1):
        LSW = LSW + math.sqrt((SWx[i + 1] - SWx[i]) ** 2 + (SWy[i + 1] - SWy[i]) ** 2)
    rSW = LeadingEdgeGrading ** (1 / NFFA)
    if LeadingEdgeGrading != 1:
        dx2SW = LeadingEdgeGrading * LSW * ((1 - rSW**NFFA) / (1 - rSW)) ** (-1)
    else:
        dx2SW = LSW / NFFA

    # North West Quadrant of Airfoil
    NWx = xTop[xTop < 0.25]
    NWy = yTop[xTop < 0.25]
    LNW = 0
    for i in range(len(NWx) - 1):
        LNW = LNW + math.sqrt((NWx[i + 1] - NWx[i]) ** 2 + (NWy[i + 1] - NWy[i]) ** 2)
    rNW = LeadingEdgeGrading ** (1 / NFFA)
    if LeadingEdgeGrading != 1:
        dx2 = LeadingEdgeGrading * LNW * ((1 - rNW**NFFA) / (1 - rNW)) ** (-1)
    else:
        dx2 = LNW / NFFA

    # North East Quadrant of Airfoil
    NEx = xTop[xTop > 0.25]
    NEy = yTop[xTop > 0.25]
    LNE = 0
    for i in range(len(NEx) - 1):
        LNE = LNE + math.sqrt((NEx[i + 1] - NEx[i]) ** 2 + (NEy[i + 1] - NEy[i]) ** 2)
    NNE = int(
        np.rint(
            math.log(TrailingEdgeGrading)
            / math.log(1 - dx2 * (1 / LNE - TrailingEdgeGrading / LNE))
        )
    )
    dxNET = TrailingEdgeGrading * dx2
    # South East Quadrant of Airfoil
    SEx = xBottom[xBottom > 0.25]
    SEy = yBottom[xBottom > 0.25]
    LSE = 0
    for i in range(len(SEx) - 1):
        LSE = LSE + math.sqrt((SEx[i + 1] - SEx[i]) ** 2 + (SEy[i + 1] - SEy[i]) ** 2)
    NSE = int(
        np.rint(
            math.log(TrailingEdgeGrading)
            / math.log(1 - dx2SW * (1 / LSE - TrailingEdgeGrading / LSE))
        )
    )
    dxSET = TrailingEdgeGrading * dx2SW

    NWake = int(
        np.rint(
            math.log(MaxCellSize / dxSET)
            / math.log(1 - dxSET / WakeLength + MaxCellSize / WakeLength)
        )
    )
    WakeGrading = MaxCellSize / dxSET
    # Calculates Normal vectors of the airfoil for extrusion of boundary layer
    nxTop = np.zeros(TopCount)
    nyTop = np.zeros(TopCount)
    for i in range(TopCount):
        if i == 0:
            nyTop[i] = (
                -(xTop[i + 1] - xTop[i])
                / ((xTop[i + 1] - xTop[i]) ** 2 + (yTop[i + 1] - yTop[i]) ** 2) ** 0.5
            )
            nxTop[i] = (yTop[i + 1] - yTop[i]) / (
                (xTop[i + 1] - xTop[i]) ** 2 + (yTop[i + 1] - yTop[i]) ** 2
            ) ** 0.5
        elif i == TopCount - 1:
            nyTop[i] = (
                -(xTop[i] - xTop[i - 1])
                / ((xTop[i] - xTop[i - 1]) ** 2 + (yTop[i] - yTop[i - 1]) ** 2) ** 0.5
            )
            nxTop[i] = (yTop[i] - yTop[i - 1]) / (
                (xTop[i] - xTop[i - 1]) ** 2 + (yTop[i] - yTop[i - 1]) ** 2
            ) ** 0.5
        else:
            nyTop[i] = (
                -(xTop[i + 1] - xTop[i - 1])
                / ((xTop[i + 1] - xTop[i - 1]) ** 2 + (yTop[i + 1] - yTop[i - 1]) ** 2)
                ** 0.5
            )
            nxTop[i] = (yTop[i + 1] - yTop[i - 1]) / (
                (xTop[i + 1] - xTop[i - 1]) ** 2 + (yTop[i + 1] - yTop[i - 1]) ** 2
            ) ** 0.5
    nxBottom = np.zeros(BottomCount)
    nyBottom = np.zeros(BottomCount)
    for i in range(BottomCount):
        if i == 0:
            nyBottom[i] = (
                -(xBottom[i + 1] - xBottom[i])
                / (
                    (xBottom[i + 1] - xBottom[i]) ** 2
                    + (yBottom[i + 1] - yBottom[i]) ** 2
                )
                ** 0.5
            )
            nxBottom[i] = (yBottom[i + 1] - yBottom[i]) / (
                (xBottom[i + 1] - xBottom[i]) ** 2 + (yBottom[i + 1] - yBottom[i]) ** 2
            ) ** 0.5
        elif i == BottomCount - 1:
            nyBottom[i] = (
                -(xBottom[i] - xBottom[i - 1])
                / (
                    (xBottom[i] - xBottom[i - 1]) ** 2
                    + (yBottom[i] - yBottom[i - 1]) ** 2
                )
                ** 0.5
            )
            nxBottom[i] = (yBottom[i] - yBottom[i - 1]) / (
                (xBottom[i] - xBottom[i - 1]) ** 2 + (yBottom[i] - yBottom[i - 1]) ** 2
            ) ** 0.5
        else:
            nyBottom[i] = (
                -(xBottom[i + 1] - xBottom[i - 1])
                / (
                    (xBottom[i + 1] - xBottom[i - 1]) ** 2
                    + (yBottom[i + 1] - yBottom[i - 1]) ** 2
                )
                ** 0.5
            )
            nxBottom[i] = (yBottom[i + 1] - yBottom[i - 1]) / (
                (xBottom[i + 1] - xBottom[i - 1]) ** 2
                + (yBottom[i + 1] - yBottom[i - 1]) ** 2
            ) ** 0.5

    # Tail Point and tail normal for kutta condition wake angle
    XT = (X[1] + X[numpoint - 2]) / 2
    YT = (Y[1] + Y[numpoint - 2]) / 2
    nxT = -(0 - YT) / ((1 - XT) ** 2 + (0 - YT) ** 2) ** (0.5)
    nyT = (1 - XT) / ((1 - XT) ** 2 + (0 - YT) ** 2) ** (0.5)
    thetawake = math.atan(nxT / nyT)

    InletGrading = LeadingEdgeGrading * DomainHeight / inletGradingFactor
    rInletGrading = InletGrading ** (1 / NFFA)
    dxInletGrading = (3.14159 * DomainHeight / 4) * (
        (1 - rInletGrading**NFFA) / (1 - rInletGrading)
    ) ** (-1)
    Ltop = 1 + math.tan(TrailingBlockAngle * 3.14159 / 180) * DomainHeight / 2

    Px = np.zeros(19)
    Py = np.zeros(19)
    Px[0] = 0
    Py[0] = 0
    Px[1] = 1 + WakeLength
    Py[1] = -DomainHeight / 2
    Px[2] = 1 + math.tan(TrailingBlockAngle * 3.14159 / 180) * DomainHeight / 2
    Py[2] = -DomainHeight / 2
    Px[3] = 0
    Py[3] = -DomainHeight / 2
    Px[4] = -DomainHeight / 2
    Py[4] = 0
    Px[5] = 0
    Py[5] = DomainHeight / 2
    Px[6] = 1 + math.tan(TrailingBlockAngle * 3.14159 / 180) * DomainHeight / 2
    Py[6] = DomainHeight / 2
    Px[7] = 1 + WakeLength
    Py[7] = DomainHeight / 2
    Px[9] = 1 + WakeLength
    Py[9] = 0 - math.tan(thetawake) * WakeLength
    Px[8] = 1 + WakeLength
    Py[8] = Py[9] + BLHeight
    Px[10] = 1 + WakeLength
    Py[10] = Py[9] - BLHeight
    Px[12] = 1
    Py[12] = 0
    Px[11] = (
        1
        + BLHeight
        * (yTop[1] - yTop[0])
        / ((xTop[1] - xTop[0]) ** 2 + (yTop[1] - yTop[0]) ** 2) ** 0.5
    )
    Py[11] = (
        0
        - BLHeight
        * (xTop[1] - xTop[0])
        / ((xTop[1] - xTop[0]) ** 2 + (yTop[1] - yTop[0]) ** 2) ** 0.5
    )
    Px[13] = (
        1
        + BLHeight
        * (yBottom[BottomCount - 1] - yBottom[BottomCount - 2])
        / (
            (xBottom[BottomCount - 1] - xBottom[BottomCount - 2]) ** 2
            + (yBottom[BottomCount - 1] - yBottom[BottomCount - 2]) ** 2
        )
        ** 0.5
    )
    Py[13] = (
        0
        - BLHeight
        * (xBottom[BottomCount - 1] - xBottom[BottomCount - 2])
        / (
            (xBottom[BottomCount - 1] - xBottom[BottomCount - 2]) ** 2
            + (yBottom[BottomCount - 1] - yBottom[BottomCount - 2]) ** 2
        )
        ** 0.5
    )
    Px[15] = 0.25
    Py[15] = np.interp(
        0.25, xTop[::-1], yTop[::-1]
    )  # [::-1] reverses list so that list is monotonically increasing with X
    ny15 = (0.001) / (
        (0.001) ** 2
        + (
            np.interp(0.251, xTop[::-1], yTop[::-1])
            - np.interp(0.25, xTop[::-1], yTop[::-1])
        )
        ** 2
    ) ** 0.5
    nx15 = (
        -(
            np.interp(0.251, xTop[::-1], yTop[::-1])
            - np.interp(0.25, xTop[::-1], yTop[::-1])
        )
        / (
            (0.001) ** 2
            + (
                np.interp(0.251, xTop[::-1], yTop[::-1])
                - np.interp(0.25, xTop[::-1], yTop[::-1])
            )
            ** 2
        )
        ** 0.5
    )
    Px[14] = 0.25 + nx15 * BLHeight
    Py[14] = Py[15] + ny15 * BLHeight
    Px[16] = 0.25
    Py[16] = np.interp(0.25, xBottom, yBottom)
    ny17 = (0.001) / (
        (0.001) ** 2
        + (
            np.interp(0.251, xBottom[::-1], yBottom[::-1])
            - np.interp(0.25, xBottom[::-1], yBottom[::-1])
        )
        ** 2
    ) ** 0.5
    nx17 = (
        -(
            np.interp(0.251, xBottom[::-1], yBottom[::-1])
            - np.interp(0.25, xBottom[::-1], yBottom[::-1])
        )
        / (
            (0.001) ** 2
            + (
                np.interp(0.251, xBottom[::-1], yBottom[::-1])
                - np.interp(0.25, xBottom[::-1], yBottom[::-1])
            )
            ** 2
        )
        ** 0.5
    )
    Px[17] = 0.25 - nx15 * BLHeight
    Py[17] = Py[16] - ny15 * BLHeight
    Px[18] = -BLHeight
    Py[18] = 0

    # Defines Control Points for meshing
    CPx = np.zeros(2)
    CPy = np.zeros(2)
    CPx[0] = -DomainHeight / 2 * math.cos(3.14159 / 4)
    CPy[0] = DomainHeight / 2 * math.sin(3.14159 / 4)
    CPx[1] = -DomainHeight / 2 * math.cos(3.14159 / 4)
    CPy[1] = -DomainHeight / 2 * math.sin(3.14159 / 4)

    NWCPx = np.zeros(len(NWx))
    NWCPy = np.zeros(len(NWx))
    for i in range(len(NWx)):
        NWCPx[i] = NWx[i] + np.interp(NWx[i], xTop[::-1], nxTop[::-1]) * BLHeight
        NWCPy[i] = NWy[i] + np.interp(NWx[i], xTop[::-1], nyTop[::-1]) * BLHeight
    LNWBL = 0
    for i in range(len(NWx) - 1):
        LNWBL += (
            (NWCPx[i + 1] - NWCPx[i]) ** 2 + (NWCPy[i + 1] - NWCPy[i]) ** 2
        ) ** 0.5

    SWCPx = np.zeros(len(SWx))
    SWCPy = np.zeros(len(SWx))
    for i in range(len(SWx)):
        SWCPx[i] = SWx[i] + np.interp(SWx[i], xBottom, nxBottom) * BLHeight
        SWCPy[i] = SWy[i] + np.interp(SWx[i], xBottom, nyBottom) * BLHeight
    LSWBL = 0
    for i in range(len(SWx) - 1):
        LSWBL += (
            (SWCPx[i + 1] - SWCPx[i]) ** 2 + (SWCPy[i + 1] - SWCPy[i]) ** 2
        ) ** 0.5

    NECPx = np.zeros(len(NEx))
    NECPy = np.zeros(len(NEx))
    for i in range(len(NEx)):
        NECPx[i] = NEx[i] + np.interp(NEx[i], xTop[::-1], nxTop[::-1]) * BLHeight
        NECPy[i] = NEy[i] + np.interp(NEx[i], xTop[::-1], nyTop[::-1]) * BLHeight
    LNEBL = 0
    for i in range(len(NEx) - 1):
        LNEBL += (
            (NECPx[i + 1] - NECPx[i]) ** 2 + (NECPy[i + 1] - NECPy[i]) ** 2
        ) ** 0.5
    rNWBL = LeadingEdgeGrading ** (1 / NFFA)
    if LeadingEdgeGrading != 1:
        dxNEBL = (LNWBL) * ((1 - rNWBL**NFFA) / (1 - rNWBL)) ** (-1)
    else:
        dxNEBL = LNWBL / NFFA

    SECPx = np.zeros(len(SEx))
    SECPy = np.zeros(len(SEx))
    for i in range(len(SEx)):
        SECPx[i] = SEx[i] + np.interp(SEx[i], xBottom, nxBottom) * BLHeight
        SECPy[i] = SEy[i] + np.interp(SEx[i], xBottom, nyBottom) * BLHeight
    LSEBL = 0
    for i in range(len(SEx) - 1):
        LSEBL += (
            (SECPx[i + 1] - SECPx[i]) ** 2 + (SECPy[i + 1] - SECPy[i]) ** 2
        ) ** 0.5
    rSWBL = LeadingEdgeGrading ** (1 / NFFA)
    if LeadingEdgeGrading != 1:
        dxSEBL = (LSWBL) * ((1 - rSWBL**NFFA) / (1 - rSWBL)) ** (-1)
    else:
        dxSEBL = LSWBL / NFFA

    # Calculations of gradings and cell-sizes

    e = 100
    rR = 1.5
    dr = 0.00001
    while e > 0.0000001:
        rRold = rR
        fR = dxInletGrading * (1 - rR**NNE) / (1 - rR) - Ltop
        ru = rR + dr
        rl = rR - dr
        fu = dxInletGrading * (1 - ru**NNE) / (1 - ru) - Ltop
        fl = dxInletGrading * (1 - rl**NNE) / (1 - rl) - Ltop
        dfdr = (fu - fl) / (2 * dr)
        rR = rRold - fR / dfdr
        e = np.abs(rR - rRold)
    topGrading = rR**NNE
    BottomGrading = topGrading

    SmallCellTop = dxInletGrading * topGrading
    LTopWake = WakeLength - math.tan(thetawake) * DomainHeight / 2

    e = 100
    rR = 1.5
    dr = 0.00001
    while e > 0.0000001:
        rRold = rR
        fR = SmallCellTop * (1 - rR**NWake) / (1 - rR) - LTopWake
        ru = rR + dr
        rl = rR - dr
        fu = SmallCellTop * (1 - ru**NWake) / (1 - ru) - LTopWake
        fl = SmallCellTop * (1 - rl**NWake) / (1 - rl) - LTopWake
        dfdr = (fu - fl) / (2 * dr)
        rR = rRold - fR / dfdr
        e = np.abs(rR - rRold)
    TopWakeGrading = rR**NWake
    e = 100
    rR = 1.5
    dr = 0.00001
    while e > 0.0000001:
        rRold = rR
        fR = dxNEBL * (1 - rR**NNE) / (1 - rR) - LNEBL
        ru = rR + dr
        rl = rR - dr
        fu = dxNEBL * (1 - ru**NNE) / (1 - ru) - LNEBL
        fl = dxNEBL * (1 - rl**NNE) / (1 - rl) - LNEBL
        dfdr = (fu - fl) / (2 * dr)
        rR = rRold - fR / dfdr
        e = np.abs(rR - rRold)

    NEBLEGrading = rR**NNE
    e = 100
    rR = 1.5
    dr = 0.00001
    while e > 0.0000001:
        rRold = rR
        fR = dxSEBL * (1 - rR**NSE) / (1 - rR) - LSEBL
        ru = rR + dr
        rl = rR - dr
        fu = dxSEBL * (1 - ru**NSE) / (1 - ru) - LSEBL
        fl = dxSEBL * (1 - rl**NSE) / (1 - rl) - LSEBL
        dfdr = (fu - fl) / (2 * dr)
        rR = rRold - fR / dfdr
        e = np.abs(rR - rRold)
    SEBLEGrading = rR**NSE

    L14_5 = math.sqrt((Px[14] - Px[5]) ** 2 + (Py[14] - Py[5]) ** 2)
    e = 100
    rR = 1.5
    dr = 0.00001
    while e > 0.0000001:
        rRold = rR
        fR = MaxLayerThickness * (1 - rR**NFF) / (1 - rR) - L14_5
        ru = rR + dr
        rl = rR - dr
        fu = MaxLayerThickness * (1 - ru**NFF) / (1 - ru) - L14_5
        fl = MaxLayerThickness * (1 - rl**NFF) / (1 - rl) - L14_5
        dfdr = (fu - fl) / (2 * dr)
        rR = rRold - fR / dfdr
        e = np.abs(rR - rRold)
    Grading14_5 = rR**NFF

    L17_3 = math.sqrt((Px[17] - Px[3]) ** 2 + (Py[17] - Py[3]) ** 2)
    e = 100
    rR = 1.5
    dr = 0.00001
    while e > 0.0000001:
        rRold = rR
        fR = MaxLayerThickness * (1 - rR**NFF) / (1 - rR) - L17_3
        ru = rR + dr
        rl = rR - dr
        fu = MaxLayerThickness * (1 - ru**NFF) / (1 - ru) - L17_3
        fl = MaxLayerThickness * (1 - rl**NFF) / (1 - rl) - L17_3
        dfdr = (fu - fl) / (2 * dr)
        rR = rRold - fR / dfdr
        e = np.abs(rR - rRold)
    Grading17_3 = rR**NFF

    # Writes blockMeshDict file into system directory of current folder
    lines = []
    lines.append(
        "/*--------------------------------*- C++ -*----------------------------------*\\ \n"
    )
    lines.append("  =========                 | \n")
    lines.append(
        "  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox \n"
    )
    lines.append("   \\    /   O peration     | Website:  https://openfoam.org \n")
    lines.append("    \\  /    A nd           | Version:  6 \n")
    lines.append("     \\/     M anipulation  | \n")
    lines.append(
        "\\*---------------------------------------------------------------------------*/ \n"
    )
    lines.append("FoamFile \n")
    lines.append("{ \n")
    lines.append("    version     2.0; \n")
    lines.append("    format      ascii; \n")
    lines.append("    class       dictionary; \n")
    lines.append("    object      blockMeshDict; \n")
    lines.append("} \n")
    lines.append(
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n"
    )
    lines.append("\n")
    lines.append(f"convertToMeters {ChordLength}; \n")
    lines.append("\n")
    lines.append("vertices\n")
    lines.append("(\n")
    for i in range(19):
        linestring = f"    ({Px[i]} {Py[i]} 0)\n"
        lines.append(linestring)
    for i in range(19):
        linestring = f"    ({Px[i]} {Py[i]} 1)\n"
        lines.append(linestring)
    lines.append(");\n\n")
    lines.append("blocks\n")
    lines.append("(\n")
    lines.append(
        f"    hex (2 1 10 13 21 20 29 32) ({NWake} {NFF} 1) simpleGrading ({TopWakeGrading} {WakeGrading} {WakeGrading} {TopWakeGrading} {1/FFGrading} {1/FFGrading} {1/FFGrading} {1/FFGrading} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (3 2 13 17 22 21 32 36) ({NSE} {NFF} 1) simpleGrading ({BottomGrading} {SEBLEGrading} {SEBLEGrading} {BottomGrading} {1/Grading17_3} {1/FFGrading} {1/FFGrading} {1/Grading17_3} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (4 3 17 18 23 22 36 37) ({NFFA} {NFF} 1) edgeGrading ({1/(InletGrading)} {1/LeadingEdgeGrading} {1/LeadingEdgeGrading} {1/(InletGrading)} {1/FFGrading} {1/Grading17_3} {1/Grading17_3} {1/FFGrading} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (4 18 14 5 23 37 33 24) ({NFF} {NFFA} 1) edgeGrading ({1/FFGrading} {1/Grading14_5} {1/Grading14_5} {1/FFGrading} {1/(InletGrading)} {1/LeadingEdgeGrading} {1/LeadingEdgeGrading} {1/(InletGrading)} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (14 11 6 5 33 30 25 24) ({NNE} {NFF} 1) edgeGrading ({NEBLEGrading} {topGrading} {topGrading} {NEBLEGrading} {Grading14_5} {FFGrading} {FFGrading} {Grading14_5} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (11 8 7 6 30 27 26 25) ({NWake} {NFF} 1) simpleGrading ({WakeGrading} {TopWakeGrading} {TopWakeGrading} {WakeGrading} {FFGrading} {FFGrading} {FFGrading} {FFGrading} 1 1 1 1) \n"
    )

    lines.append(
        f"    hex (12 9 8 11 31 28 27 30) ({NWake} {NBL} 1) simpleGrading ({WakeGrading} {BLGrading} 1) \n"
    )
    lines.append(
        f"    hex (15 12 11 14 34 31 30 33) ({NNE} {NBL} 1) egeeGrading ({TrailingEdgeGrading} {NEBLEGrading} {NEBLEGrading} {TrailingEdgeGrading} {BLGrading} {BLGrading} {BLGrading} {BLGrading} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (0 15 14 18 19 34 33 37) ({NFFA} {NBL} 1) edgeGrading ({LeadingEdgeGrading} {1/LeadingEdgeGrading} {1/LeadingEdgeGrading} {LeadingEdgeGrading} {BLGrading} {BLGrading} {BLGrading} {BLGrading} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (18 17 16 0 37 36 35 19) ({NFFA} {NBL} 1) edgeGrading ({1/LeadingEdgeGrading} {LeadingEdgeGrading} {LeadingEdgeGrading} {1/LeadingEdgeGrading} {1/BLGrading} {1/BLGrading} {1/BLGrading} {1/BLGrading} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (17 13 12 16 36 32 31 35) ({NSE} {NBL} 1) edgeGrading ({SEBLEGrading} {TrailingEdgeGrading} {TrailingEdgeGrading} {SEBLEGrading} {1/BLGrading} {1/BLGrading} {1/BLGrading} {1/BLGrading} 1 1 1 1) \n"
    )
    lines.append(
        f"    hex (13 10 9 12 32 29 28 31) ({NWake} {NBL} 1) simpleGrading ({WakeGrading} {1/BLGrading} 1) \n"
    )
    lines.append(");\n\n")
    lines.append("edges\n")
    lines.append("(\n")
    #################
    # DOMAIN POINTS #
    #################
    lines.append("    polyLine 11 14\n")
    lines.append("    (\n")
    for i in range(len(NEx)):
        lines.append(f"        ({NECPx[i]} {NECPy[i]} 0)\n")
    lines.append("    )\n")
    lines.append("    polyLine 30 33\n")
    lines.append("    (\n")
    for i in range(len(NEx)):
        lines.append(f"        ({NECPx[i]} {NECPy[i]} 1)\n")
    lines.append("    )\n")
    lines.append("    polyLine 14 18\n")
    lines.append("    (\n")
    for i in range(len(NWx)):
        lines.append(f"        ({NWCPx[i]} {NWCPy[i]} 0)\n")
    lines.append("    )\n")
    lines.append("    polyLine 33 37\n")
    lines.append("    (\n")
    for i in range(len(NWx)):
        lines.append(f"        ({NWCPx[i]} {NWCPy[i]} 1)\n")
    lines.append("    )\n")
    lines.append("    polyLine 18 17\n")
    lines.append("    (\n")
    for i in range(len(SWx)):
        lines.append(f"        ({SWCPx[i]} {SWCPy[i]} 0)\n")
    lines.append("    )\n")
    lines.append("    polyLine 37 36\n")
    lines.append("    (\n")
    for i in range(len(SWx)):
        lines.append(f"        ({SWCPx[i]} {SWCPy[i]} 1)\n")
    lines.append("    )\n")
    lines.append("    polyLine 17 13\n")
    lines.append("    (\n")
    for i in range(len(SEx)):
        lines.append(f"        ({SECPx[i]} {SECPy[i]} 0)\n")
    lines.append("    )\n")
    lines.append("    polyLine 36 32\n")
    lines.append("    (\n")
    for i in range(len(SEx)):
        lines.append(f"        ({SECPx[i]} {SECPy[i]} 1)\n")
    lines.append("    )\n")
    lines.append("    spline 12 15\n")
    lines.append("    (\n")
    for i in range(len(NEx)):
        lines.append(f"        ({NEx[i]} {NEy[i]} 0)\n")
    lines.append("    )\n")
    lines.append("    spline 31 34\n")
    lines.append("    (\n")
    for i in range(len(NEx)):
        lines.append(f"        ({NEx[i]} {NEy[i]} 1)\n")
    lines.append("    )\n")
    lines.append(f"    arc 4 5 ({CPx[0]} {CPy[0]} 0)\n")
    lines.append(f"    arc 3 4 ({CPx[1]} {CPy[1]} 0)\n")
    lines.append(f"    arc 23 24 ({CPx[0]} {CPy[0]} 1)\n")
    lines.append(f"    arc 22 23 ({CPx[1]} {CPy[1]} 1)\n")

    ################################
    # AIRFOIL INTERPOLATION POINTS #
    ################################
    lines.append("    spline 15 0\n")
    lines.append("    (\n")
    for i in range(len(NWx)):
        lines.append(f"        ({NWx[i]} {NWy[i]} 0)\n")
    lines.append("    )\n")
    lines.append("    spline 34 19\n")
    lines.append("    (\n")
    for i in range(len(NWx)):
        lines.append(f"        ({NWx[i]} {NWy[i]} 1)\n")
    lines.append("    )\n")
    lines.append("    spline 0 16\n")
    lines.append("    (\n")
    for i in range(len(SWx)):
        lines.append(f"        ({SWx[i]} {SWy[i]} 0)\n")
    lines.append("    )\n")
    lines.append("    spline 19 35\n")
    lines.append("    (\n")
    for i in range(len(SWx)):
        lines.append(f"        ({SWx[i]} {SWy[i]} 1)\n")
    lines.append("    )\n")
    lines.append("    spline 16 12\n")
    lines.append("    (\n")
    for i in range(len(SEx)):
        lines.append(f"        ({SEx[i]} {SEy[i]} 0)\n")
    lines.append("    )\n")
    lines.append("    spline 35 31\n")
    lines.append("    (\n")
    for i in range(len(SEx)):
        lines.append(f"        ({SEx[i]} {SEy[i]} 1)\n")
    lines.append("    )\n")
    lines.append(");\n\n")

    lines.append("boundary\n")
    lines.append("(\n")
    lines.append("    frontAndBack\n")
    lines.append("    {\n")
    lines.append("        type empty;\n")
    lines.append("        faces\n")
    lines.append("        (\n")
    lines.append("          (1 2 13 10)\n")
    lines.append("          (2 3 17 13)\n")
    lines.append("          (3 4 18 17)\n")
    lines.append("          (18 4 5 14)\n")
    lines.append("          (11 14 5 6)\n")
    lines.append("          (8 11 6 7)\n")
    lines.append("          (9 12 11 8)\n")
    lines.append("          (12 15 14 11)\n")
    lines.append("          (0 18 14 15)\n")
    lines.append("          (17 18 0 16)\n")
    lines.append("          (13 17 16 12)\n")
    lines.append("          (10 13 12 9)\n")
    lines.append("          (21 20 29 32)\n")
    lines.append("          (22 21 32 36)\n")
    lines.append("          (23 22 36 37)\n")
    lines.append("          (23 37 33 24)\n")
    lines.append("          (33 30 25 24)\n")
    lines.append("          (30 27 26 25)\n")
    lines.append("          (31 28 27 30)\n")
    lines.append("          (34 31 30 33)\n")
    lines.append("          (37 19 34 33)\n")
    lines.append("          (37 36 35 19)\n")
    lines.append("          (36 32 31 35)\n")
    lines.append("          (32 29 28 31)\n")
    lines.append("         );\n")
    lines.append("     }\n")
    lines.append("    farfield\n")
    lines.append("    {\n")
    lines.append("        type patch;\n")
    lines.append("        faces\n")
    lines.append("        (\n")
    lines.append("          (4 23 24 5)\n")
    lines.append("          (5 24 25 6)\n")
    lines.append("          (6 25 26 7)\n")
    lines.append("          (27 8 7 26)\n")
    lines.append("          (28 9 8 27)\n")
    lines.append("          (29 10 9 28)\n")
    lines.append("          (20 1 10 29)\n")
    lines.append("          (21 2 1 20)\n")
    lines.append("          (22 3 2 21)\n")
    lines.append("          (23 4 3 22)\n")
    lines.append("         );\n")
    lines.append("     }\n")
    lines.append("    airfoil\n")
    lines.append("    {\n")
    lines.append("        type wall;\n")
    lines.append("        faces\n")
    lines.append("        (\n")
    lines.append("          (34 15 12 31)\n")
    lines.append("          (19 0 15 34)\n")
    lines.append("          (0 19 35 16)\n")
    lines.append("          (16 35 31 12)\n")
    lines.append("         );\n")
    lines.append("     }\n")
    lines.append(");\n")

    path = run_path / "system"
    path.mkdir(exist_ok=True, parents=True)
    with open(path / "blockMeshDict", "w") as f:
        f.writelines(lines)
