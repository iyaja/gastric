import jax.numpy as jnp
import numpy as np
import time
from scipy.spatial import cKDTree as KDTree
import itertools
import matplotlib.pyplot as plt

from jax_md import space, energy, simulate, force


import time 

from functools import partial
import numpy as onp

import jax.numpy as np

from jax import jit
from jax import grad
from jax import vmap
from jax import value_and_grad

from jax import random
from jax import lax

from jax.experimental import stax
from jax.experimental import optimizers

from jax.config import config
config.update('jax_enable_x64', True)

# NOTE(schsam): We need this in OSS I think.
from IPython.display import HTML, display
import time

def ProgressIter(iter_fun, iter_len=0):
  if not iter_len:
    iter_len = len(iter_fun)
  out = display(progress(0, iter_len), display_id=True)
  for i, it in enumerate(iter_fun):
    yield it
    out.update(progress(i + 1, iter_len))

def progress(value, max):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 45%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
  
sns.set_style(style='white')
sns.set(font_scale=1.6)

def format_plot(x, y):  
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)
  
def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()

f32 = np.float32
f64 = np.float64

def draw_system(R, box_size, marker_size, color=None):
  if color == None:
    color = [64 / 256] * 3
  ms = marker_size / box_size

  R = onp.array(R)

  marker_style = dict(
      linestyle='none', 
      markeredgewidth=3,
      marker='o', 
      markersize=ms, 
      color=color, 
      fillstyle='none')

  plt.plot(R[:, 0], R[:, 1], **marker_style)
  plt.plot(R[:, 0] + box_size, R[:, 1], **marker_style)
  plt.plot(R[:, 0], R[:, 1] + box_size, **marker_style)
  plt.plot(R[:, 0] + box_size, R[:, 1] + box_size, **marker_style)
  plt.plot(R[:, 0] - box_size, R[:, 1], **marker_style)
  plt.plot(R[:, 0], R[:, 1] - box_size, **marker_style)
  plt.plot(R[:, 0] - box_size, R[:, 1] - box_size, **marker_style)

  plt.xlim([0, box_size])
  plt.ylim([0, box_size])
  plt.axis('off')

def square_lattice(N, box_size):
  Nx = int(np.sqrt(N))
  Ny, ragged = divmod(N, Nx)
  if Ny != Nx or ragged:
    assert ValueError('Particle count should be a square. Found {}.'.format(N))
  length_scale = box_size / Nx
  R = []
  for i in range(Nx):
    for j in range(Ny):
      R.append([i * length_scale, j * length_scale])
  return np.array(R)




N = 64
Lbox = 8
L = Lbox
T = 2
mass = 1.0
dt = 0.003
rc = 2.5
sigma = 1.123  # diameter of the particles
V = Lbox ** 3


def kinetic_energy(m, v):
    """
    args:
        m (float): mass of particles
        v (array): velocities of particles,
        assumed to be a 2D array of shape (N, 3)
    returns:
        float: total kinetic energy
    """
    total_vsq = jnp.einsum("ij,ij", v, v)
    return 0.5 * m * total_vsq


def my_pos_in_box(pos, lbox):
    """ wrap positions inside simulation box

    Args:
    pos (jnp.array): positions, shape (natom, ndim)
    lbox (float): box side length
    Returns:
    jnp.array: pos in box
    """
    N = len(pos)
    L = lbox
    pos = pos - 0.5 * L
    a = pos // L
    b = pos % L
    pos_in_box = b - 0.5 * L
    return pos_in_box


def potential_energy(pos, rc, L):
    """
    args:
        dist (array): distance table with shape (N, N)
            i.e. dist[i,j] is the distance
            between particle i and particle j
            in the minimum image convention
            note that the diagonal of dist can be zero
        rc (float): cutoff distance for interaction
            i.e. if dist[i,j] > rc, the pair potential between
            i and j will be 0
        L: box length 
    returns:
        float: total potential energy
    """
    # dist = scipy.spatial.distance.pdist(pos)
    # dist = jnp.where(dist > 0.5*L, dist-L, jnp.where(dist < -0.5*L, dist+L , dist))

    table = pos[:, jnp.newaxis, :] - pos[jnp.newaxis, :, :]
    dist_vec = minimum_image(table, L)
    # dist_vec[jnp.diag_indices(len(dist_vec))] = jnp.inf
    dist_vec = dist_vec.at[jnp.diag_indices(len(dist_vec))].set(jnp.inf)
    r = jnp.linalg.norm(dist_vec, axis=-1)

    v = 4 * ((1 / r) ** 12 - (1 / r) ** 6)
    vc = 4 * ((1 / rc) ** 12 - (1 / rc) ** 6)
    # v[r < rc] -= vc  # shift
    # v[r >= rc] = 0  # cut
    v = v.at[r < rc].set(v[r < rc] - vc)
    v = v.at[r >= rc].set(0)
    return 0.5 * jnp.sum(v)


def minimum_image(r, L):
    """
    args:
        r : array of any shape
        L : box size
    returns:
        array of the same shape as r,
        the minimum image of r
    """
    return r - L * jnp.round(r / L)


def initial_positions(N, radius, L):
    """ 
    Simple helper function to generate fcc positions 
    
    args: 
        N (int): number of particles 
        L (float): side length of cubic box
        radius (float): distance on lattice 
    returns:
        array of shape (N,3) with particle positions on fcc lattice 
     """
    a = radius * np.sqrt(2) * 2
    S = list(np.arange(-L / 2.0, L / 2.0 - a, a))

    FCC = list(itertools.product(S, S, S))
    FCC = np.vstack(
        (
            FCC,
            FCC + a / 2.0 * np.array([1, 1, 0]),
            FCC + a / 2.0 * np.array([1, 0, 1]),
            FCC + a / 2.0 * np.array([0, 1, 1]),
        )
    )

    q = np.random.choice(len(FCC), N, replace=False)
    return FCC[q]


def initial_velocities(N, m, T):
    """
    initialize velocities at a desired temperature
    required for: initialization

    args:
        N (int): number of particles
        m (float): mass of particles
        T (float): desired temperature
    returns:
        array: initial velocities, with shape (N, 3)
    """
    velocities = np.random.rand(N, 3) - 0.5
    # zero the total net velocity
    velocities -= jnp.average(velocities, axis=0)
    # correct temperature
    current_temp = 2 * kinetic_energy(m, velocities) / (3 * N)
    velocities *= jnp.sqrt(T / current_temp)
    return velocities


def velocity_verlet(force_func, pos, vel, mass, dt, rc, L):
    """
    advance system according to velocity verlet

    args:
        force_func: a function that calculates all forces between all particles 
        pos (array): coordinates of particles
        vel (array): velocities of particles
        mass (float): mass of particles
        dt (float): timestep by which to advance

        rc (float): cutoff
        L (float): length of cubic box
    returns:
        array, array: new positions, new velocities
    """

    accel = force_func(pos, rc, L) / mass
    # move
    vel_half = vel + 0.5 * dt * accel
    pos_new = pos + dt * vel_half
    pos_new = minimum_image(pos_new, L)

    # repeat force calculation for new pos
    accel = force_func(pos_new, rc, L) / mass
    # finish move
    vel_new = vel_half + 0.5 * dt * accel

    return pos_new, vel_new


# above is to visually present the initial condition


def force_smarter(pos, rc, L):
    """
    calculate forces for all particles in system 
    args:
        pos (array): all positions, shape (N,3)
        rc (float): cutoff distance for interaction
        i.e. if dist[i,j] > rc, particle i will feel no force
        from particle j
        L (float): box size 
    returns:
        array: forces f on all particles, with shape (N, 3)
        i.e. f[3,0] gives the force on particle i
        in the x direction
    """
    table = pos[:, jnp.newaxis, :] - pos[jnp.newaxis, :, :]
    dist_vec = minimum_image(table, L)
    dist_vec.at[jnp.diag_indices(len(dist_vec))].set(jnp.inf)
    r_len = jnp.linalg.norm(dist_vec, axis=-1)
    one_over_r_squared = 1.0 / pow(r_len, 2)
    one_over_r_six = pow(one_over_r_squared, 3)
    magnitude = 24.0 * one_over_r_six * one_over_r_squared * (2.0 * one_over_r_six - 1)
    magnitude.at[r_len >= rc].set(0)
    val = magnitude[:, :, jnp.newaxis] * dist_vec
    val = jnp.nansum(val, axis=1)

    return val


def displacement_table(coordinates, L):
    """
    required for: force(), advance()
    args:
        coordinates (array): coordinates of particles,
        assumed to have shape (N, 3)
        e.g. coordinates[3,0] should give the x component
        of particle 3
        L (float): side length of cubic box,
        must be known in order to compute minimum image
    returns:
        array: table of displacements r
        such that r[i,j] is the minimum image of
        coordinates[i] - coordinates[j]
    """
    # coordinates=jnp.array(coordinates)
    coordinates = my_pos_in_box(coordinates, L)
    # print(coordinates)
    # coordinates=coordinates.tolist()
    N = len(coordinates)
    r = np.zeros((N, N, 3))

    for i in range(N):
        for j in range(N):
            r[i][j][0] = coordinates[i][0] - coordinates[j][0]
            r[i][j][1] = coordinates[i][1] - coordinates[j][1]
            r[i][j][2] = coordinates[i][2] - coordinates[j][2]
        r[i] = minimum_image(r[i], L)
    # print(r)
    return r


def eventmap(pos, vel, sigma):
    """
    this function calculate the time of each event happens
    ijnput: pos: Nx3, position of all particles
    vel: Nx3, velocities of all particles
    return: t: NxN, event map
    
    """
    N = len(pos)  # number of particles
    r = jnp.zeros((N, N, 3))  # r is the displacement of all particles
    v = jnp.zeros((N, N, 3))  # v is relative velocity of all particles
    t = jnp.zeros((N, N))
    # r[i,j]=ri-rj, a vector
    for i in range(N):
        for j in range(N):
            r[i][j] = jnp.subtract(pos[i], pos[j])
            v[i][j] = jnp.subtract(vel[i], vel[j])
            vij = jnp.linalg.norm(v[i][j])  # norm of vij
            rij = jnp.linalg.norm(r[i][j])  # norm of rij
            delta = 2 * jnp.dot(r[i][j], v[i][j]) ** 2 - 4 * vij ** 2 * (
                rij ** 2 - sigma ** 2
            )
            if delta >= 0:
                t[i][j] = (-2 * jnp.dot(r[i][j], v[i][j]) + jnp.sqrt(delta)) / (
                    2 * vij ** 2
                )
            else:
                t[i][j] = 5000
                # assign a large number
    return t


def pressure(pos, force, V, T):
    N = len(pos)
    P = N * T / V
    for i in range(N):
        P = P + jnp.dot(pos[i], force[i]) / (3 * V)
    return P


def proce_part(pos, vel, i, j, mint):
    """
    this function returns pos and vel after particle collision
    """
    pos = pos + vel * mint
    rij_vec = pos[i] - pos[j]
    rij = jnp.linalg.norm(rij_vec)  # norm of rij_vec
    vij_vec = vel[i] - vel[j]
    vel[i] = vel[i] - (jnp.dot(rij_vec, vij_vec) / rij) * rij_vec / rij
    vel[j] = vel[j] + (jnp.dot(rij_vec, vij_vec) / rij) * rij_vec / rij
    # t_pass=(mint)
    return pos, vel


def advance(pos, vel, sigma, L, force_smarter, mass, dt, rc):
    N = len(pos)
    disp = displacement_table(pos, L)
    dist = jnp.linalg.norm(disp, axis=-1)
    close_coor = jnp.asarray(
        jnp.where((0 < dist) & (dist < sigma))
    ).T  # the atoms that are about to collide
    if len(close_coor) != 0:
        # print('yeah')
        [i, j] = close_coor[0]
        # calculate the vel after collision
        rij_vec = disp[i, j]

        rij = dist[i, j]  # norm of rij_vec

        vij_vec = vel[i] - vel[j]
        vel.at[i].add(-(jnp.dot(rij_vec, vij_vec) / rij) * rij_vec / rij)
        vel.at[j].add(vel[j] + (jnp.dot(rij_vec, vij_vec) / rij) * rij_vec / rij)
        # jnp.delete(close_coor, 0)
        # jnp.delete(close_coor, -1) # diagonal symmetric
    pos, vel = velocity_verlet(force_smarter, pos, vel, mass, dt, rc, L)

    return pos, vel


def main():
    """pos = initial_positions(N,0.5,Lbox)
    vel = initial_velocities(N,mass,T)
    colli=3

    tmap=eventmap(pos, vel, sigma)
    print(tmap)
    [pos1,vel1]=advance(colli, pos, vel, sigma)
    print(pos1)   
    print(pos)
    """

    pos = initial_positions(N, 0.5, Lbox)
    vel = initial_velocities(N, mass, T)

    p = potential_energy(pos, rc, Lbox)
    k = kinetic_energy(mass, vel)

    print("Initial potential energy ", p)
    print("Initial kinetic energy ", k)

    PE = []
    KE = []
    start = time.time()
    Nsteps = 2000
    P = []
    for i in range(Nsteps):

        pos, vel = advance(pos, vel, sigma, L, force_smarter, mass, dt, rc)
        # print(pos)
        PE.append(potential_energy(pos, rc, Lbox))
        KE.append(kinetic_energy(mass, vel))
        force = force_smarter(pos, rc, L)
        pres = pressure(pos, force, V, T)
        P.append(pres)
    end = time.time()
    total_time = end - start
    print(
        "elapsed time: %3.2f seconds, performance %3.2f steps/second"
        % (total_time, Nsteps / total_time)
    )

    # print(jnp.add(PE,KE))
    plt.figure()
    plt.plot(KE)
    plt.figure()
    plt.plot(P)
    plt.xlabel("steps")
    plt.ylabel("pressure")
    pltitle = "T=", T, " L=", L, " N=", N
    plt.title(pltitle)

    print(jnp.average(P[1000::]))


if __name__ == "__main__":
    main()
