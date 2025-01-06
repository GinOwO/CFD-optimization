"""
Python3 code based on
    Airfoil generation using CST parameterization method
    Version 1.0.0.0 (1.67 KB) by Pramudita Satria Palar
    Generates an airfoil shape using CST method 
    https://mathworks.com/matlabcentral/fileexchange/42239-airfoil-generation-using-cst-parameterization-method

Original License:
-----------------------------------------------------------------------------------
Copyright (c) 2013, Pramudita Satria Palar
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from math import factorial
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import math


def cst2coords(wl: List[float], wu: List[float], dz: float, N: int) -> np.ndarray:
    def CST_airfoil(wl: List[float], wu: List[float], dz: float, N: int) -> np.ndarray:
        """
        Create a set of airfoil coordinates using the CST parametrization method.

        Args:
            wl (List[float]): CST weights for the lower surface.
            wu (List[float]): CST weights for the upper surface.
            dz (float): Trailing edge thickness.
            N (int): Number of discretization points.

        Returns:
            np.ndarray: A 2D array of x and y coordinates of the airfoil.
        """
        zeta = np.linspace(0, 2 * np.pi, N + 1)
        x = 0.5 * (np.cos(zeta) + 1)
        N1, N2 = 0.5, 1.0
        zerind = np.argmax(x == 0)
        yl = ClassShape(wl, x[:zerind], N1, N2, -dz)
        yu = ClassShape(wu, x[zerind:], N1, N2, dz)
        y = np.concatenate([yl, yu])
        coord = np.column_stack((x, y))

        return coord

    def ClassShape(
        w: List[float], x: np.ndarray, N1: float, N2: float, dz: float
    ) -> np.ndarray:
        """
        Calculate the class and shape function for CST parametrization.

        Args:
            w (List[float]): CST weights.
            x (np.ndarray): x-coordinates.
            N1 (float): Class function parameter N1.
            N2 (float): Class function parameter N2.
            dz (float): Trailing edge displacement.

        Returns:
            np.ndarray: y-coordinates of the airfoil.
        """
        C = x**N1 * (1 - x) ** N2
        n = len(w) - 1
        K = [factorial(n) / (factorial(i) * factorial(n - i)) for i in range(n + 1)]

        S = np.zeros_like(x)
        for i, xi in enumerate(x):
            S[i] = sum(
                w[j] * K[j] * (xi**j) * ((1 - xi) ** (n - j)) for j in range(n + 1)
            )
        y = C * S + x * dz
        return y

    return CST_airfoil(wl, wu, dz, N)


if __name__ == "__main__":
    wl = [float(i) for i in input(f"Enter wl(space-separated): ").split()]
    wu = [float(i) for i in input(f"Enter wu(space-separated): ").split()]
    dz = float(input("Enter dz: "))
    N = int(input("Enter N: "))
    coords = cst2coords(wl, wu, dz, N)

    print(coords)

    arr = np.array(coords)
    plt.plot(arr[:, 0], arr[:, 1])
    plt.axis("equal")
    plt.show()
