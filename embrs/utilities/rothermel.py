from embrs.utilities.fuel_models import Anderson13
import numpy as np

def calc_propagation_in_cell(fuel: Anderson13, spread_directions, wind_speed_m_s, wind_dir_deg, slope_angle_deg, slope_dir_deg):
    
    wind_speed_ft_min = 196.85 * wind_speed_m_s

    if slope_angle_deg == 0:
        rel_wind_dir_deg = 0

    else:
        rel_wind_dir_deg = wind_dir_deg - slope_dir_deg
        if rel_wind_dir_deg < 0:
            rel_wind_dir_deg += 360

    rel_wind_dir = np.deg2rad(rel_wind_dir_deg)
    slope_angle = np.deg2rad(slope_angle_deg)
    slope_dir = np.deg2rad(slope_dir_deg)
    spread_directions = np.deg2rad(spread_directions)

    output = []
    for decomp_dir in spread_directions:
        # rate of spread along gamma in ft/min, fireline intensity along gamma in Btu/ft/min
        r_gamma, I_gamma = calc_r_and_i_along_dir(fuel, decomp_dir, wind_speed_ft_min, rel_wind_dir, slope_angle, slope_dir)

        r_gamma /= 196.85 # convert to m/s
        I_gamma *= 0.05767 # convert to kW/m # TODO: double check this conversion

        output.append((r_gamma, I_gamma))

    return output

def calc_r_0(I_r, flux_ratio, heat_sink):
    R_0 = (I_r * flux_ratio)/heat_sink

    return R_0

def calc_I_r(net_fuel_load, sav_ratio, rel_packing_ratio, heat_content, moist_damping, mineral_damping):

    A = 133 * sav_ratio ** (-0.7913)

    max_reaction_vel = (sav_ratio ** 1.5) * (495 + 0.0594 * sav_ratio ** 1.5) ** (-1)
    opt_reaction_vel = max_reaction_vel * (rel_packing_ratio ** A) * np.exp(A*(1-rel_packing_ratio))

    I_r = opt_reaction_vel * net_fuel_load * heat_content * moist_damping * mineral_damping

    return I_r

def calc_flux_ratio(sav_ratio, rho_b):

    packing_ratio = rho_b / 32    
    flux_ratio = (192 + 0.2595*sav_ratio)**(-1) * np.exp((0.792 + 0.681*sav_ratio**0.5)*(packing_ratio + 0.1))

    return flux_ratio

def calc_heat_sink(rho_b, sav_ratio, m_f):

    epsilon = np.exp(-138/sav_ratio)
    Q_ig = 250 + 1116 * m_f

    heat_sink = rho_b * epsilon * Q_ig

    return heat_sink

def calc_r_and_i_along_dir(fuel: Anderson13, decomp_dir, wind_speed, wind_dir, slope_angle, slope_dir):
    """Calculates the rate of spread in direction gamma from the ignition point

    :param gamma: _description_
    :type gamma: _type_
    :return: _description_
    :rtype: _type_
    """

    sav_ratio = fuel.sav_ratio
    net_fuel_load = fuel.net_fuel_load
    rel_packing_ratio = fuel.rel_packing_ratio
    heat_content = fuel.heat_content
    rho_b = fuel.rho_b
    m_f = fuel.fuel_moisture # TODO: need to figure out how we're handling this
    m_x = fuel.m_x

    E, B, C = calc_E_B_C(sav_ratio)

    moist_damping = calc_moisture_damping(m_f, m_x)
    mineral_damping = calc_mineral_damping()

    I_r = calc_I_r(net_fuel_load, sav_ratio, rel_packing_ratio, heat_content, moist_damping, mineral_damping)

    flux_ratio = calc_flux_ratio(sav_ratio, rho_b)
    heat_sink = calc_heat_sink(rho_b, sav_ratio, m_f)

    R_0 = calc_r_0(I_r, flux_ratio, heat_sink)

    phi_w = calc_wind_factor(wind_speed, rel_packing_ratio, E, B, C)
    phi_s = calc_slope_factor(slope_angle, rho_b)

    R_h, alpha = calc_r_h(R_0, phi_w, phi_s, wind_dir)

    # TODO: Double Check that this is correct
    gamma = abs((alpha + slope_dir) - decomp_dir) % (2*np.pi)
    gamma = np.min([gamma, 2*np.pi - gamma])

    phi_e = calc_effective_wind_factor(R_h, R_0)
    u_e = calc_effective_wind_speed(phi_e, rel_packing_ratio, E, B, C)

    e = calc_eccentricity(u_e)

    R_gamma = R_h * ((1 - e)/(1 - e * np.cos(gamma)))

    t_r = 384 / sav_ratio
    H_a = I_r * t_r
    I_gamma = H_a * R_gamma

    return R_gamma, I_gamma

def calc_E_B_C(sav_ratio):
    E = 0.715 * np.exp(-3.59e-4 * sav_ratio)
    B = 0.02526 * sav_ratio ** 0.54
    C = 7.47 * np.exp(-0.133 * sav_ratio**0.55)

    return E, B, C

def calc_wind_factor(wind_speed, rel_packing_ratio, E, B, C):
    phi_w = C * (wind_speed ** B) * rel_packing_ratio ** (-E)

    return phi_w

def calc_slope_factor(phi, rho_b):
    packing_ratio = rho_b / 32

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

def calc_effective_wind_speed(phi_e, rel_packing_ratio, E, B, C):
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
    u_e = (((phi_e * rel_packing_ratio**E)/C) ** (1/B))

    return u_e

def calc_eccentricity(u_e):
    u_e_mph = u_e * 0.0113636

    z = 1 + 0.25 * u_e_mph
    e = ((z**2 - 1)**0.5)/z

    return e

def calc_r_h(R_0, phi_w, phi_s, omega):
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

    t = 60

    d_w = R_0 * phi_w * t
    d_s = R_0 * phi_s * t

    x = d_s + d_w * np.cos(omega)
    y = d_w * np.sin(omega)

    D_h = np.sqrt(x**2 + y**2)

    R_h = R_0 + (D_h / t)

    alpha = np.arcsin(y/D_h) # TODO: check that this is ok without abs(y)

    return R_h, alpha

