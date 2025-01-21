from embrs.utilities.fuel_models import Anderson13
from embrs.fire_simulator.cell import Cell
import numpy as np

def calc_propagation_in_cell(cell, wind_speed_m_s, wind_dir_deg, R_h_in = None):
    wind_speed_ft_min = 196.85 * wind_speed_m_s

    slope_angle_deg = cell.slope_deg
    slope_dir_deg = cell.aspect

    if slope_angle_deg == 0:
        rel_wind_dir_deg = 0
        cell.aspect = wind_dir_deg

    else:
        rel_wind_dir_deg = wind_dir_deg - slope_dir_deg
        if rel_wind_dir_deg < 0:
            rel_wind_dir_deg += 360

    rel_wind_dir = np.deg2rad(rel_wind_dir_deg)
    spread_directions = np.deg2rad(cell.directions)

    R_h, R_0, I_r, alpha = calc_r_h(cell, wind_speed_ft_min, slope_angle_deg, rel_wind_dir)

    if R_h_in is not None:
        R_h = R_h_in

    e = calc_eccentricity(cell.fuel_type, R_h, R_0)

    r_list = []
    I_list = []
    for decomp_dir in spread_directions:
        # rate of spread along gamma in ft/min, fireline intensity along gamma in Btu/ft/min
        r_gamma, I_gamma = calc_r_and_i_along_dir(cell, decomp_dir, R_h, I_r, alpha, e)

        r_gamma /= 196.85 # convert to m/s
        I_gamma *= 0.05767 # convert to kW/m # TODO: double check this conversion

        r_list.append(r_gamma)
        I_list.append(I_gamma)

    return np.array(r_list), np.array(I_list)

def calc_r_0(fuel, m_f):

    flux_ratio = calc_flux_ratio(fuel)
    I_r = calc_I_r(fuel, m_f)
    heat_sink = calc_heat_sink(fuel, m_f)

    R_0 = (I_r * flux_ratio)/heat_sink

    return R_0, I_r

def calc_I_r(fuel, m_f):

    moist_damping = calc_moisture_damping(m_f, fuel.m_x)
    mineral_damping = calc_mineral_damping()

    A = 133 * fuel.sav_ratio ** (-0.7913)

    max_reaction_vel = (fuel.sav_ratio ** 1.5) * (495 + 0.0594 * fuel.sav_ratio ** 1.5) ** (-1)
    opt_reaction_vel = max_reaction_vel * (fuel.rel_packing_ratio ** A) * np.exp(A*(1-fuel.rel_packing_ratio))

    I_r = opt_reaction_vel * fuel.net_fuel_load * fuel.heat_content * moist_damping * mineral_damping

    return I_r

def calc_flux_ratio(fuel):
    rho_b = fuel.rho_b
    sav_ratio = fuel.sav_ratio

    packing_ratio = rho_b / 32    
    flux_ratio = (192 + 0.2595*sav_ratio)**(-1) * np.exp((0.792 + 0.681*sav_ratio**0.5)*(packing_ratio + 0.1))

    return flux_ratio

def calc_heat_sink(fuel, m_f):

    rho_b = fuel.rho_b
    sav_ratio = fuel.sav_ratio

    epsilon = np.exp(-138/sav_ratio)
    Q_ig = 250 + 1116 * m_f

    heat_sink = rho_b * epsilon * Q_ig

    return heat_sink

def calc_r_and_i_along_dir(cell: Cell, decomp_dir, R_h, I_r, alpha, e):
    """Calculates the rate of spread in direction gamma from the ignition point

    :param gamma: _description_
    :type gamma: _type_
    :return: _description_
    :rtype: _type_
    """
    fuel = cell.fuel_type
    slope_dir = np.deg2rad(cell.aspect)

    gamma = abs((alpha + slope_dir) - decomp_dir) % (2*np.pi)
    gamma = np.min([gamma, 2*np.pi - gamma])

    R_gamma = R_h * ((1 - e)/(1 - e * np.cos(gamma)))

    t_r = 384 / fuel.sav_ratio # Residence time
    H_a = I_r * t_r
    I_gamma = H_a * R_gamma

    return R_gamma, I_gamma

def calc_E_B_C(fuel):
    sav_ratio = fuel.sav_ratio

    E = 0.715 * np.exp(-3.59e-4 * sav_ratio)
    B = 0.02526 * sav_ratio ** 0.54
    C = 7.47 * np.exp(-0.133 * sav_ratio**0.55)

    return E, B, C

def calc_wind_factor(fuel, wind_speed):

    E, B, C = calc_E_B_C(fuel)
    phi_w = C * (wind_speed ** B) * fuel.rel_packing_ratio ** (-E)

    return phi_w

def calc_slope_factor(fuel, phi):
    packing_ratio = fuel.rho_b / 32

    phi_s = 5.275 * (packing_ratio ** (-0.3)) * (np.tan(phi)) ** 2

    return phi_s


def calc_moisture_damping(m_f, m_x):
    r_m = m_f / m_x

    moist_damping = 1 - 2.59 * r_m + 5.11 * (r_m)**2 - 3.52 * (r_m)**3

    return moist_damping

def calc_mineral_damping(s_e = 0.010):

    mineral_damping = 0.174 * s_e ** (-0.19)

    return mineral_damping


def calc_effective_wind_factor(R_h, R_0):
    """Effective wind factor in direction of maximum spread

    :param R_h: _description_
    :type R_h: _type_
    :param R_0: _description_
    :type R_0: _type_
    :return: _description_
    :rtype: _type_
    """
    phi_e = (R_h / R_0) - 1

    return phi_e

def calc_effective_wind_speed(fuel, R_h, R_0):
    """_summary_

    :param phi_e: _description_
    :type phi_e: _type_
    :param rel_packing_ratio: _description_
    :type rel_packing_ratio: _type_
    :param E: _description_
    :type E: _type_
    :param B: _description_
    :type B: _type_
    :param C: _description_
    :type C: _type_
    :return: _description_
    :rtype: _type_
    """

    E, B, C = calc_E_B_C(fuel)
    phi_e = calc_effective_wind_factor(R_h, R_0)


    u_e = (((phi_e * fuel.rel_packing_ratio**E)/C) ** (1/B))

    return u_e

def calc_eccentricity(fuel, R_h, R_0):

    u_e = calc_effective_wind_speed(fuel, R_h, R_0)

    u_e_mph = u_e * 0.0113636

    z = 1 + 0.25 * u_e_mph
    e = ((z**2 - 1)**0.5)/z

    return e

def calc_r_h(cell, wind_speed, slope_angle, omega, R_0 = None, I_r = None):
    """Calculate the rate of spread in the direction of maximum spread, heading fire

    :param R_0: _description_
    :type R_0: _type_
    :param phi_w: _description_
    :type phi_w: _type_
    :param phi_s: _description_
    :type phi_s: _type_
    :param omega: _description_
    :type omega: _type_
    :return: _description_
    :rtype: _type_
    """

    fuel = cell.fuel_type
    m_f = cell.dead_m
    
    if R_0 is None or I_r is None:
        R_0, I_r = calc_r_0(fuel, m_f)

    phi_w = calc_wind_factor(fuel, wind_speed)
    phi_s = calc_slope_factor(fuel, slope_angle)

    t = 60

    d_w = R_0 * phi_w * t
    d_s = R_0 * phi_s * t

    x = d_s + d_w * np.cos(omega)
    y = d_w * np.sin(omega)

    D_h = np.sqrt(x**2 + y**2)

    R_h = R_0 + (D_h / t)

    alpha = np.arcsin(y/D_h) # TODO: check that this is ok without abs(y)

    return R_h, R_0, I_r, alpha

