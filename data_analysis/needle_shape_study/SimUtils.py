import numpy as np
import matplotlib.pyplot as plt
import time 
from numba import jit, njit, prange
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba.typed import List

def R_cones(z,needle_spacing,diameter):
    '''
    params:
    z: coordinate z of the point of interest ( along trap z axis)
    needle_spacing: needle tip to tip distance
    diameter: diameter of the needle at the end of the z_value i.e rod end diameter
    
    returns:
    R: the radius of the cone for given z value
    '''
    z_0 = needle_spacing * 0.5
    if np.abs(z) < z_0 :
        return 0
    if np.abs(z) >= z_0 : 
        return 0.5 * diameter/(xlen - z_0) * (np.abs(z) - z_0)

def R_parabolic(z,needle_spacing,diameter):
    '''
    diameter: the diameter of the needle at the end of the z_value'''
    z_0 = needle_spacing * 0.5
    if np.abs(z) < z_0 :
        return 0
    if np.abs(z) >= z_0 :
        return 0.5 * diameter/((xlen - z_0)**0.5) * (np.abs(z)-z_0)**0.5 
    
def Tip_rad_cur(needle_spacing,diameter):
    '''
    Returns:
    R: radius of curvature for a parabolic tip defined as per R_parabolic
    see https://en.wikipedia.org/wiki/Radius_of_curvature for reference (2D formula)
    for a function y = a * x**(1/2), R = a**2 / 2 at tip i.e at x = 0
    '''
    z_0 = needle_spacing * 0.5
    return (0.5 * diameter/((xlen - z_0)**0.5))**2 / 2

def Dia_giv_rad_cur(needle_spacing,rad_cur):
    z_0 = needle_spacing * 0.5
    return ( 8 * rad_cur * (xlen - z_0 ) )**0.5


def draw_cones(X,Y,needle_spacing,diameter):
    '''
    writes 1 to the points where conical needles are located
    '''
    cones = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (np.abs(X[i,j]) > (needle_spacing*0.5)) & (np.abs(Y[i,j]) <= np.abs(R_cones(X[i,j],needle_spacing,diameter))) :
                cones[i,j] = 1
    return cones      


def draw_parabolic(X,Y,needle_spacing,diameter):
    '''
    writes 1 to the points where parabolic needles are located

    '''
    cones = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (np.abs(X[i,j]) > (needle_spacing*0.5)) & (np.abs(Y[i,j]) <= np.abs(R_parabolic(X[i,j],needle_spacing,diameter))) :
                cones[i,j] = 1
    return cones

def draw_parabolic_single(X,Y,needle_spacing,diameter):
    
    cones = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (X[i,j] > (needle_spacing*0.5)) & (np.abs(Y[i,j]) <= np.abs(R_parabolic(X[i,j],needle_spacing,diameter))) :
                cones[i,j] = 1
    return cones

def draw_parabolic_asym(X,Y,needle_spacing,diameter,asym_factor):
    '''
    asym_factor: difference between the diameter of the two needles at the rod end
    '''
    cones = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] > 0:
                if (np.abs(X[i,j]) > (needle_spacing*0.5)) & (np.abs(Y[i,j]) <= np.abs(R_parabolic(X[i,j],needle_spacing,diameter))) :
                    cones[i,j] = 1
            if X[i,j] < 0:
                if (np.abs(X[i,j]) > (needle_spacing*0.5)) & (np.abs(Y[i,j]) <= np.abs(R_parabolic(X[i,j],needle_spacing,diameter+asym_factor))) :
                    cones[i,j] = 1
    return cones


def set_grid_points(blades):
    grid_points = List()
    for i in range(0,blades.shape[0]-1):
        for j in range(0,blades.shape[1]-1):
            if blades[i,j] == 0 :            
                grid_points.append((i,j))
    return grid_points


@jit(nopython=True)
def fast_numb_solve_test(blades,gp,iterations,tol):
    #largest_update = 0
    for iteration in range(0,iterations):
        largest_update = 0
        for grid_point in gp:
            i = grid_point[0]
            j = grid_point[1]
            value = 0.25 * (blades[i+1,j] + blades[i-1,j] + blades[i,j+1] + blades[i,j-1])
            update = np.abs(value - blades[i,j])
            blades[i,j] = value 
            if update > largest_update : 
                largest_update = update
        if largest_update < tol:
            break
    print(iteration)
    print(largest_update)
    return blades



q = 1.60217662*(10**(-19))
m = 2.8395214559472*10**(-25)



def fit_quadratic(z,V_0,K):
    return V_0 + 0.5 * K * z**2

def trapfreq(k):
    return np.sqrt(np.abs(k)/m)/(2*np.pi)



def solve_parabolic(X,Y,needle_spacing,diameter):
    needle_parabolic = draw_parabolic(X,Y,needle_spacing,diameter)
    grid_points_parabolic = set_grid_points(needle_parabolic)
    parabolic_solved = fast_numb_solve_test(needle_parabolic,grid_points_parabolic,100000,1e-6)
    roi = 15
    z_center = (X.shape[1]-1)//2
    z_data = X[150,z_center-roi:z_center+roi+1] * 1e-3
    v_data = parabolic_solved[150,z_center-roi:z_center+roi+1] * q
    popt, pconv = curve_fit(fit_quadratic,z_data,v_data)
    
    return trapfreq(popt[1])


def solve_cone(X,Y,needle_spacing,diameter):
    needle_cone = draw_cones(X,Y,needle_spacing,diameter)
    grid_points_cones = set_grid_points(needle_cone)
    cone_solved = fast_numb_solve_test(needle_cone,grid_points_cones,100000,1e-6)
    roi = 15
    z_center = (X.shape[1]-1)//2
    z_data = X[150,z_center-roi:z_center+roi+1] * 1e-3
    v_data = cone_solved[150,z_center-roi:z_center+roi+1] * q
    popt, pconv = curve_fit(fit_quadratic,z_data,v_data)
    return trapfreq(popt[1])