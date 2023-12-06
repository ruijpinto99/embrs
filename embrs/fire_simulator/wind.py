"""Wind representation that defines the wind conditions for a simulation and calculates the effect
on fire propagation

.. autoclass:: Wind
    :members:
"""

from typing import Tuple
import numpy as np

from embrs.utilities.fire_util import WindAdjustments
from embrs.fire_simulator.cell import Cell

class Wind:
    """Wind class that specifies the wind conditions for a given simulation.

    :param forecast: list of tuples representing the wind vector at equally spaced instants in
                     time. Each element represents (speed (m/s), direction (deg)) at an instant.
    :type forecast: list
    :param time_step: time step between each point in 'forecast' (in minutes)
    :type time_step: float
    """

    def __init__(self, forecast: list, time_step: float):
        """Constructor method for a wind instance, creates an initial wind vector and saves input
        parameters.
        """
        print("Configuring wind...")
        wind_dir_rad = forecast[0][1] * (np.pi/180)
        wind_mag_m_s = forecast[0][0]
        self._vec = [wind_mag_m_s*np.cos(wind_dir_rad), wind_mag_m_s*np.sin(wind_dir_rad)]
        self._wind_speed = wind_mag_m_s
        self._wind_dir_deg = forecast[0][1]
        self._time_step = time_step
        self._forecast = forecast
        self._curr_index = 0

    def _update_wind(self, curr_time_s: int) -> bool:
        """Updates the wind to reflect the forecasted conditions at the inputted time.

        :param curr_time_s: current sim time (in seconds).
        :type curr_time_s: int
        """

        curr_time_m = curr_time_s / 60
        curr_index = int(np.floor(curr_time_m / self.time_step))

        if curr_index != self._curr_index:
            self._curr_index = curr_index
            self._wind_speed = self.forecast[self.curr_index][0]
            self._wind_dir_deg = self.forecast[self.curr_index][1]
            wind_dir_rad = self.wind_dir_deg * (np.pi/180)
            self._vec = [self.wind_speed*np.cos(wind_dir_rad),self.wind_speed*np.sin(wind_dir_rad)]

            return True
        return False

    def _calc_wind_effect(self, curr_cell: Cell, disp: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate the effect the wind has on the spread probability from 'cell' to one of its
        neighbors at a displacement 'disp' away.

        :param curr_cell: :class:`~fire_simulator.fire.Cell` instance that is currently on fire
        :type curr_cell: :class:`~fire_simulator.fire.Cell`
        :param disp: tuple representing the delta in col, row from 'curr_cell' to one of its
                     neighbors
        :type disp: Tuple[float, float]
        :return: a tuple of form (alpha_w, k_w). 
        
        - 'alpha_w' is used to weight the probability of ignition.
        - `k_w' is used to weight the propagation velocity of the fire.
        :rtype: Tuple[float, float]
        """
        disp_vec = curr_cell.mapping[disp]

        dot_proj = sum(a*b for a, b in zip(disp_vec, self.vec))

        k_w = max(0, np.exp(0.1783*dot_proj) - 0.486)

        if self.wind_speed == 0:
            alpha_w = 1

        else:
            cos_rel_angle = dot_proj / (np.linalg.norm(disp_vec) * np.linalg.norm(self.vec))
            rel_angle = np.arccos(cos_rel_angle)
            rel_angle = np.degrees(rel_angle)
            adj_vel_kmh = max(self.wind_speed * 3.6 - 8, 0)
            alpha_w = self.interpolate_wind_adjustment(adj_vel_kmh, rel_angle)

        return alpha_w, k_w

    def interpolate_parameters(self, wind_speed: float) -> Tuple[float, float, float]:
        """Calculates the parametes for the Lorentzian function based on the input wind speed.
        Interpolates a series of Lorentzian fits to determine the parameters.

        :param wind_speed: wind speed in km/h to find parameters for
        :type wind_speed: float
        :return: Peak amplitude (A), Full width at half maximum (gamma), and constant offset (C)
                 of the Lorentzian
        :rtype: Tuple[float, float, float]
        """

        param_mapping = WindAdjustments.wind_speed_param_mapping

        sorted_speeds = sorted(param_mapping.keys())

        if wind_speed in sorted_speeds:
            return param_mapping[wind_speed]

        # Find two wind speeds closest to the given wind speed for interpolation
        v_lower = max([s for s in sorted_speeds if s <= wind_speed])
        v_upper = min([s for s in sorted_speeds if s >= wind_speed])

        # Extract parameters for the two closest wind speeds
        A_lower, gamma_lower, C_lower = param_mapping[v_lower]
        A_upper, gamma_upper, C_upper = param_mapping[v_upper]

        # Calculate weights for interpolation
        w1 = (v_upper - wind_speed) / (v_upper - v_lower)
        w2 = 1 - w1

        # Interpolate the parameters
        A_interp = w1 * A_lower + w2 * A_upper
        gamma_interp = w1 * gamma_lower + w2 * gamma_upper
        C_interp = w1 * C_lower + w2 * C_upper

        return A_interp, gamma_interp, C_interp

    def lorentzian(self, x: float, A: float, gamma: float, C: float) -> float:
        """Calculate the output of a Lorentzian function with the given input parameters.

            :param x: Point to calculate the output of the Lorentzian for, representing the
                      relative angle in degrees.
            :type x: float
            :param A: Peak amplitude of the Lorentzian peak.
            :type A: float
            :param gamma: Scale parameter related to the full width at half maximum (FWHM) of the
                          peak.
            :type gamma: float
            :param C: Constant baseline offset of the Lorentzian peak.
            :type C: float
            :return: The value of the Lorentzian function at point x.
            :rtype: float
        """
        return A / (1 + (x/gamma)** 2) + C

    def interpolate_wind_adjustment(self, wind_speed_kmh: float, direction: float) -> float:
        """Calculates the alpha_w parameter which models the probability adjustment due to the
        wind. Interpolates a series of lorentzian functions that define the adjustment factor.

        :param wind_speed_kmh: wind speed in km/h
        :type wind_speed_kmh: float
        :param direction: direction of wind in degrees relative to the propagation direction being
                          considered
        :type direction: float
        :return: wind adjustment factor, alpha_w
        :rtype: float
        """

        # Interpolate the parameters for the given wind speed
        A, gamma, C = self.interpolate_parameters(wind_speed_kmh)

        # Calculate the interpolated value using the Lorentzian function
        return self.lorentzian(direction, A, gamma, C)

    @property
    def vec(self) -> list:
        """Current wind conditions as a 2D vector [x velocity (m/s), y velocity (m/s)].
        """
        return self._vec

    @property
    def wind_speed(self) -> float:
        """Current speed of the wind (m/s).
        """
        return self._wind_speed

    @property
    def wind_dir_deg(self) -> float:
        """Current direction of the wind (deg).
        """
        return self._wind_dir_deg

    @property
    def time_step(self) -> float:
        """Time step between each point in the wind's 'forecast' (in minutes).
        """
        return self._time_step

    @property
    def forecast(self) -> list:
        """List of tuples representing the wind vector at equally spaced instants in
        time. 
        
        Each element represents (speed (m/s), direction (deg)) at an instant. 
        """
        return self._forecast

    @property
    def curr_index(self) -> int:
        """Current index the wind is on within its forecast.
        """
        return self._curr_index
