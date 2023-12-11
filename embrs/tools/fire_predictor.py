"""Module for running a prediction on a fire to project its state in a set amount of time.

Runs a simplified version of the fire simulator to provide predictions.

Utilizes a homogenous fuel map, adds noise to the wind forecast, and has the ability to be biased
towards a more or less conservative prediction.

.. autoclass:: FirePredictor
    :members:

"""
from embrs.fire_simulator.fire import FireSim
from embrs.fire_simulator.wind import Wind
from embrs.base_classes.base_fire import BaseFireSim
from embrs.utilities.fire_util import FireTypes, CellStates, UtilFuncs, ControlledBurnParams
from embrs.utilities.action import *

import numpy as np
import heapq

class FirePredictor(BaseFireSim):
    """Predictor class responsible for running predictions over a fixed time horizon.

    :param orig_fire: FireSim object to predict the future spread for.
    :type orig_fire: FireSim
    :param time_horizon_hr: Time horizon in hours for which the model should predict over.
    :type time_horizon_hr: float
    :param fuel_type: Fuel type to be used for the predictor's homogenous fuel map. If -1, the
        dominant fuel type in the original map will be used, defaults to -1
    :type fuel_type: int, optional
    :param bias: Bias term that controls how conservative the model is, will tend to over-predict
        if >1, under-predict if <1, defaults to 1
    :type bias: float, optional
    :param time_step_s: Time step in seconds used for the prediction. This will be the temporal 
        granularity of prediction result. If None, the time step of orig_fire will be used.
    :type time_step_s: float
    :param cell_size_m: Cell size in meters used for the prediction. This will be the spatial
        granularity of the prediction result. If None, the cell size of orig_fire will be used.
    :type cell_size_m: float, defaults to None
    """
    def __init__(self, orig_fire: FireSim, time_horizon_hr: float, fuel_type: int = -1, bias: float = 1, time_step_s: float = None, cell_size_m: float = None):
        """Constructor for prediction model.
        """

        if time_step_s is None:
            time_step_s = orig_fire.time_step * 2

        if cell_size_m is None:
            cell_size_m = orig_fire.cell_size * 2

        if not isinstance(fuel_type, int) or fuel_type <= 0 or fuel_type > 13:
            fuel_type = UtilFuncs.get_dominant_fuel_type(orig_fire.base_fuel_map)

            if fuel_type != -1:
                orig_fire.logger.log_message("Invalid fuel type passed to prediction model, defaulted to dominant fuel type in original map")
                print("Invalid fuel type passed to prediction model, defaulted to dominant fuel type in original map")

        self.bias = bias

        fuel_map = np.full(orig_fire.base_fuel_map.shape, fuel_type)
        fuel_res = orig_fire.fuel_res

        topography_map = orig_fire.base_topography
        topography_res = orig_fire.topography_res

        wind_forecast = self.generate_noisy_wind(orig_fire.wind_vec._forecast)
        wind_t_step = orig_fire.wind_vec._time_step

        wind_vec = Wind(wind_forecast, wind_t_step)

        roads = orig_fire.roads
        fire_breaks = orig_fire.fire_breaks

        time_horizon_s = time_horizon_hr * 60 * 60

        # extract initial ignition from curr_fires of orig_fire
        initial_ignition = UtilFuncs.get_cell_polygons(orig_fire.curr_fires)

        sim_size = orig_fire.size
        burnt_cells = UtilFuncs.get_cell_polygons(orig_fire._burnt_cells)

        display_freq_s = orig_fire.display_frequency

        super().__init__(fuel_map, fuel_res, topography_map,
                topography_res, wind_vec, roads, fire_breaks,
                time_step_s, cell_size_m, time_horizon_s, initial_ignition,
                sim_size, burnt_cells, display_freq_s)

        self.start_time_s = orig_fire.curr_time_s
        self._curr_time_s = self.start_time_s

        self.action_heap = []
        self._future_fires = {}
        self._reduced_fuel = {}

    def generate_noisy_wind(self, wind_forecast: list) -> list:
        """Adds noise to the true wind forecast using a auto-regressive model.

        :param wind_forecast: Wind forecast being used by the original fire simulation
        :type wind_forecast: list
        :return: New forecast with noise added
        :rtype: list
        """
        phi = 0.8

        new_forecast = []
        speed_errors = np.zeros(len(wind_forecast))
        dir_errors = np.zeros(len(wind_forecast))

        for i, (speed, dir) in enumerate(wind_forecast):
            if i != 0:
                speed_errors[i] = speed_errors[i - 1] * phi + np.random.normal(0, 1)
                dir_errors[i] = dir_errors[i - 1] * phi + np.random.normal(0, 2.5)

            new_forecast.append((speed + speed_errors[i], dir + dir_errors[i]))

        return new_forecast

    def iterate(self):
        """Iterate the fire prediction one time-step forward.
        """
        self._init_iteration()

        # Update wind
        self.wind_changed = self._wind_vec._update_wind(self._curr_time_s)

        if len(self.action_heap) != 0:
            self.perform_actions()

        # Iterate over currently burning cells
        for curr_cell in self._curr_fires:

            if len(curr_cell.burnable_neighbors) == 0:
                self._curr_fires_anti_cache.append(curr_cell)
                continue

            # iterate over burnable neighbors
            neighbors_to_remove = []
            for neighbor_id, (dx, dy) in curr_cell.burnable_neighbors:
                neighbor = self._cell_dict[neighbor_id]

                neighbor_burnable = self.check_if_burnable(curr_cell, neighbor)

                if not neighbor_burnable:
                    if neighbor.fire_type != FireTypes.PRESCRIBED:
                        neighbors_to_remove.append((neighbor_id, (dx, dy)))

                else:
                    prob, v_prop = self._calc_prob(curr_cell, neighbor, (dx, dy))

                    prob *= self.bias
                    v_prop *= self.bias

                    if np.random.random() < prob:

                        time_entry = self._future_fires.get(self.curr_time_s)

                        if not time_entry:
                            self._future_fires[self.curr_time_s] = [(neighbor.x_pos, neighbor.y_pos)]

                        else:
                            time_entry.append((neighbor.x_pos, neighbor.y_pos))

                        neighbor._set_vprop(v_prop)
                        self.set_state_at_cell(neighbor, CellStates.FIRE, curr_cell.fire_type)

                        # Add cell to prediction dictionary
                        if neighbor.fire_type == FireTypes.PRESCRIBED:
                            self._reduced_fuel[self._curr_time_s].append((neighbor.x_pos, neighbor.y_pos))

                        else:    
                            self._future_fires[self._curr_time_s].append((neighbor.x_pos, neighbor.y_pos))

            if curr_cell.fire_type == FireTypes.WILD:
                curr_cell.burnable_neighbors.difference_update(neighbors_to_remove)
            
            else:
                curr_cell._set_fuel_content(curr_cell.fuel_content*ControlledBurnParams.burnout_fuel_frac)

        self._iters += 1

    def _init_iteration(self):
        """Set up an iteration. Resets relevant data structures for keeping track of fires burning
        on the frontier.
        """
        # Update current time
        self._curr_time_s = self.start_time_s + (self.time_step * self.iters)

        self._curr_fires = self._curr_fires | set(self._curr_fires_cache)
        self._curr_fires.difference_update(self._curr_fires_anti_cache)

        self._curr_fires_cache = []
        self._curr_fires_anti_cache = []

        if len(self._curr_fires) == 0 or (self.time_step * self.iters) >= self._sim_duration:
            self._finished = True

    def perform_actions(self):
        # Get the closest time step

        print(f"action time: {self.action_heap[0][0]}")
        print(f"sim time: {self.curr_time_s}")

        if self.action_heap[0][0] >= self.curr_time_s:
            action = heapq.heappop(self.action_heap)[1]
            action.perform(self)
            print("performing action")


    def generate_action_sequence(self):
        action_sequence = []

        for i in range(10):
            action = SetFuelContent(self.curr_time_s + 3600, i * 100, 100 * i, 0.1)
            action_sequence.append(action)
        
            action = SetFuelMoisture(self.curr_time_s + 5400, i * 150, 150 * i, 0.5)
            action_sequence.append(action)

            action = SetIgnition(self.curr_time_s + 1800, i * 200, 200 * i, FireTypes.PRESCRIBED)
            action_sequence.append(action)
    
        return action_sequence

    def run_prediction(self, action_sequence:list = None) -> dict:
        """Run a prediction

        :param action_sequence: TODO: FILL IN THIS INPUT
        :return: Dictionary where each key is a time-step and each value is a list of predicted
        ignition locations (x, y) at that time-step. Time-steps start at the time-step the original
            fire when input to the prediction model.
        :rtype: dict
        """

        self.action_heap = []

        action_sequence = self.generate_action_sequence()

        if action_sequence is not None:
            for action in action_sequence:
                heapq.heappush(self.action_heap, (action.time, action))

        while not self.finished:
            self.iterate()

        return self._future_fires

    @property
    def reduced_fuel_prediction(self) -> dict:
        """Get the regions predicted to be partially burnt by prescribed burning

        :return: Dictionary where each key is a time-step and each value is a list of predicted
        reduced fuel locations (x, y) at that time-step. Time-steps start at the time-step the
        original fire when input to the prediction model.
        :rtype: dict
        """

        return self._reduced_fuel
