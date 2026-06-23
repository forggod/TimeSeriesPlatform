import numpy as np
from .base import BaseAttractor


class HenonAttractor(BaseAttractor):
    name = "henon"

    def generate(self, steps, a=1.4, b=0.3):

        def henon(x, y, a, b):

            xn = 1 - a * x**2 + y
            yn = b * x

            return xn, yn

        data = []

        hx = np.zeros(steps + 1)
        hy = np.zeros(steps + 1)

        # начальные условия
        hx[0], hy[0] = (0.1, 0.1)
        data.append((hx[0], hy[0]))

        for i in range(steps):
            hx[i + 1], hy[i + 1] = henon(
                hx[i],
                hy[i],
                a,
                b
            )
            data.append((hx[i+1], hy[i+1]))

        return np.array(data)

    def params_schema(self):
        return {
            "steps": {
                "type": "int",
                "default": 1000
            },
            "a": {
                "type": "float",
                "default": 1.4
            },
            "b": {
                "type": "float",
                "default": 0.3
            },
        }
