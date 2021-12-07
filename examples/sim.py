from jax_md import spaces, energy, force, simulate


# Global variables
N = 64
BOX_SIZE = 8
L = BOX_SIZE
T = 2
mass = 1.0
dt = 0.003
rc = 2.5
sigma = 1.123  
V = BOX_SIZE ** 3


displacement, shift = spaces.periodic(BOX_SIZE)

