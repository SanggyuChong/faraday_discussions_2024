from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time
import ase.io
from mace.calculators import MACECalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.build import bulk, make_supercell
from  ase.md.npt import NPT
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter


frame = ase.io.read("begin.xyz") ## an arbitrary starting configuration that is already coarse-grained


frame.info = {}

calculator = MACECalculator(model_paths='./CG_water.model', device='cuda')
frame.set_calculator(calculator)

MaxwellBoltzmannDistribution(frame, 300 * units.kB)

dyn = Langevin(frame, 0.5*units.fs, temperature_K=300, friction=5e-3, logfile="log.out")

def write_frame():
        dyn.atoms.write('test.xyz', append=True)

dyn.attach(write_frame, interval=100)

dyn.run(2000000)
print("MD finished!")
