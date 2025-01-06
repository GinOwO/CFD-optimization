import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


class NACACoords:
    def __init__(self, path: str):
        x, y = [], []
        with open(path, "r") as f:
            for line in f:
                coords = line.split()
                if not coords:
                    continue
                x.append(float(coords[0]))
                y.append(float(coords[1]))
        self.m_coords = np.column_stack((x, y))

    def get_coords(self) -> np.ndarray:
        return self.m_coords

    def get_x(self) -> np.ndarray:
        return self.m_coords[0]

    def get_y(self) -> np.ndarray:
        return self.m_coords[1]

    def is_normalized(self) -> bool:
        x, y = self.get_coords()
        return all(0 <= i <= 1 and 0 <= j <= 1 for i, j in zip(x, y))

    def normalize(self) -> None:
        if self.is_normalized():
            return
        x, y = self.get_coords()
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        self.m_coords = np.array([x, y])

    def save(self, path: str) -> None:
        x, y = self.get_coords()
        df = pd.DataFrame({"x": x, "y": y})
        df.to_csv(path, index=False)

    def plot(self) -> None:
        x, y = self.get_coords()
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label="NACA")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("NACA Airfoil")
        plt.axis("equal")
        plt.legend()
        plt.grid()
        plt.show()

    def smooth(self, n: int) -> None:
        x, y = self.get_coords()
        tck, _ = splprep([x, y], s=0)
        u_new = np.linspace(0, 1, n)
        x_new, y_new = splev(u_new, tck)
        self.m_coords = np.array([x_new, y_new])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    naca = NACACoords(input_path)
    # naca.smooth(1000)
    # naca.normalize()
    # naca.save(output_path)
    naca.plot()
