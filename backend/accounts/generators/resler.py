import numpy as np
from .base import BaseAttractor


class ReslerAttractor(BaseAttractor):

    name = "resler"

    def generate(self, steps, a=0.1, b=0.1, c=14, dt=0.01):
        def resler(x, y, z, a=0.1, b=0.1, c=14):
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)

            return dx, dy, dz

        data = []

        x = np.zeros(steps + 1)
        y = np.zeros(steps + 1)
        z = np.zeros(steps + 1)

        # Задаем начальное возмущение системы
        x[0], y[0], z[0] = (0.1, 0.1, 0.1)
        data.append((x[0], y[0], z[0]))

        for i in range(steps):
            dx, dy, dz = resler(x[i], y[i], z[i])

            x[i+1] = x[i] + (dx * dt)
            y[i+1] = y[i] + (dy * dt)
            z[i+1] = z[i] + (dz * dt)

            data.append((x[i+1], y[i+1], z[i+1]))

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
            "a": {
                "type": "float",
                "default": 0.1
            },
            "b": {
                "type": "float",
                "default": 0.1
            },
            "c": {
                "type": "float",
                "default": 14
            }
        }
