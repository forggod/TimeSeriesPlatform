from .lorenz import LorenzAttractor
from .resler import ReslerAttractor
from .henon import HenonAttractor

ATTRACTORS = {
    LorenzAttractor.name: LorenzAttractor(),
    ReslerAttractor.name: ReslerAttractor(),
    HenonAttractor.name: HenonAttractor()
}
