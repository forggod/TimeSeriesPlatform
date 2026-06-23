import numpy as np
from .base import BaseAttractor


class LorenzAttractor(BaseAttractor):

    name = "lorenz"

    def generate(self, steps=1000, dt=0.01, sigma=10, rho=28, beta=8/3):
        x, y, z = 0.1, 0.0, 0.0

        data = []

        for _ in range(steps):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z

            x += dx * dt
            y += dy * dt
            z += dz * dt

            data.append((x, y, z))

        return np.array(data)

    def params_schema(self):
        return {
            "steps": {
                "type": "int",
                "default": 1000
            },
            "dt": {
                "type": "float",
                "default": 0.01
            },
            "sigma": {
                "type": "float",
                "default": 10
            },
            "rho": {
                "type": "float",
                "default": 28
            },
            "beta": {
                "type": "float",
                "default": 8/3
            }
        }
