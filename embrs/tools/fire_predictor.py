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
from embrs.utilities.fire_util import FireTypes, CellStates, UtilFuncs, ControlledBurnParams, cell_type, action_type
from embrs.utilities.action import *
from embrs.utilities.fire_util import RoadConstants as rc

from shapely.geometry import Point

import cProfile
import pstats

import numpy as np
import heapq
import copy
import mmap
import subprocess
import struct
import os

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
        # Define the path to the bin folder
        self.bin_folder = os.path.join(os.path.dirname(__file__), '..', 'bin')
        os.makedirs(self.bin_folder, exist_ok=True)  # Create the bin directory if it doesn't exist
        
        # Define file paths
        self.cell_filename = os.path.join(self.bin_folder, 'cells.dat')
        self.action_filename = os.path.join(self.bin_folder, 'actions.dat')
        self.output_path = os.path.join(self.bin_folder, 'prediction.bin')

        if time_step_s is None:
            time_step_s = orig_fire.time_step * 2

        self._time_step = time_step_s
        self._cell_size = cell_size_m

        if not isinstance(fuel_type, int) or fuel_type <= 0 or fuel_type > 13:
            if fuel_type != -1:
                orig_fire.logger.log_message("Invalid fuel type passed to prediction model, defaulted to dominant fuel type in original map")
                print("Invalid fuel type passed to prediction model, defaulted to dominant fuel type in original map")

            fuel_type = UtilFuncs.get_dominant_fuel_type(orig_fire.base_fuel_map)

        self.bias = bias

        topography_map = orig_fire.base_topography
        topography_res = orig_fire.topography_res

        wind_forecast = self.generate_noisy_wind(orig_fire.wind_vec._forecast)
        wind_t_step = orig_fire.wind_vec._time_step

        self.wind_forecast = wind_forecast
        self.wind_t_step = wind_t_step

        self.wind_forecast_str = ' '.join(f"{u} {v}" for u, v in self.wind_forecast)

        roads = orig_fire.roads
        fire_breaks = orig_fire.fire_breaks

        self.time_horizon_hr = time_horizon_hr

        # extract initial ignition from curr_fires of orig_fire
        initial_ignition = UtilFuncs.get_cell_polygons(orig_fire.curr_fires)

        sim_size = orig_fire.size
        burnt_cells = UtilFuncs.get_cell_polygons(orig_fire._burnt_cells)


        num_cols = int(np.floor(sim_size[0]/(np.sqrt(3)*cell_size_m)))
        num_rows = int(np.floor(sim_size[1]/(1.5*cell_size_m)))

        self._shape = (num_rows, num_cols)

        self._cell_grid = np.zeros((num_rows, num_cols), dtype=cell_type)

        # Populate cell_grid with cells
        id = 0
        for i in range(num_cols):
            for j in range(num_rows):
                # Initialize cell object
                self._cell_grid[j, i]['id'] = id
                
                self._cell_grid[j, i]['state'] = CellStates.FUEL # TODO: define states in pyhton
                self._cell_grid[j, i]['indices']['i'] = j # TODO: check indices
                self._cell_grid[j, i]['indices']['j'] = i

                if j % 2 == 0:
                    x_pos = i * cell_size_m * np.sqrt(3)
                else: 
                    x_pos = (i + 0.5) * cell_size_m * np.sqrt(3)

                y_pos = j * cell_size_m * 1.5

                self._cell_grid[j, i]['position']['x'] = x_pos
                self._cell_grid[j, i]['position']['y'] = y_pos
                self._cell_grid[j, i]['fuelType'] = fuel_type
                self._cell_grid[j, i]['fuelContent'] = 1.0
                self._cell_grid[j, i]['changed'] = False

                # Set cell elevation from topography map
                top_col = int(np.floor(x_pos/topography_res))
                top_row = int(np.floor(y_pos/topography_res))
                self._cell_grid[j, i]['position']['z'] = topography_map[top_row, top_col]

                # increment id counter
                id +=1

        # Set initial ignitions
        for polygon in initial_ignition:
            minx, miny, maxx, maxy = polygon.bounds

            # Get row and col indices for bounding box
            min_row = int(miny // (cell_size_m * 1.5))
            max_row = int(maxy // (cell_size_m * 1.5))
            min_col = int(minx // (cell_size_m * np.sqrt(3)))
            max_col = int(maxx // (cell_size_m * np.sqrt(3)))

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:

                        x_pos = self._cell_grid[row, col]['position']['x']
                        y_pos = self._cell_grid[row, col]['position']['y']

                        if polygon.contains(Point(x_pos, y_pos)) and fuel_type <= 13:
                            self._cell_grid[row, col]['state'] = CellStates.FIRE # TODO: define states in python

        if burnt_cells is not None:
            for polygon in burnt_cells:
                minx, miny, maxx, maxy = polygon.bounds

                # Get row and col indices for bounding box
                min_row = int(miny // (cell_size_m * 1.5))
                max_row = int(maxy // (cell_size_m * 1.5))
                min_col = int(minx // (cell_size_m * np.sqrt(3)))
                max_col = int(maxx // (cell_size_m * np.sqrt(3)))

                for row in range(min_row, max_row + 1):
                    for col in range(min_col, max_col + 1):
                        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                            
                            x_pos = self._cell_grid[row, col]['position']['x']
                            y_pos = self._cell_grid[row, col]['position']['y']
                            
                            if polygon.contains(Point(x_pos, y_pos)):
                                self._cell_grid[row, col]['state'] = CellStates.BURNT # TODO: define states in python

        # Apply fire breaks
        for fire_break in fire_breaks:
            line = fire_break['geometry']
            fuel_val = fire_break['fuel_value']
            length = line.length

            step_size = 0.5
            num_steps = int(length/step_size) + 1

            for i in range(num_steps):
                point = line.interpolate(i * step_size)

                cell = self.get_cell_from_xy(point.x, point.y, oob_ok = True)

                if cell is not None:
                    cell['fuelContent'] = fuel_val/100.0
                    
        # Apply roads
        if roads is not None:
            for road in roads:
                for point in road[0]:
                    road_x, road_y = point[0], point[1]

                    road_cell = self.get_cell_from_xy(road_x, road_y, oob_ok = True)

                    if road_cell is not None:
                        
                        if road_cell['state'] == CellStates.FIRE:
                            road_cell['state'] = CellStates.FUEL

                        road_cell['fuelContent'] = rc.road_fuel_vals[road[1]] # TODO: check that this and fire break operation changes value in cell_grid


        # Write cell data to binary file
        file_size = self._cell_grid.nbytes

        # Ensure the file is sized correctly
        with open(self.cell_filename, 'wb') as f:
            f.seek(file_size - 1)
            f.write(b'\x00')

        # Map the file and write data
        with open(self.cell_filename, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_WRITE)
            mm.write(self._cell_grid.tobytes())
            mm.close()

    def run_prediction_batch(self, batch_size, action_seq_batch = None, preprocess_actions:bool = False) -> dict:
        if action_seq_batch is not None:
            actions, total_actions = self.convert_to_action_type(action_seq_batch)

            # Write actions to memory-mapped file
            file_size = actions.nbytes

            # Ensure the file is sized correctly
            with open(self.action_filename, 'wb') as f:
                f.seek(file_size - 1)
                f.write(b'\x00')

            # Map the file and write data
            with open(self.action_filename, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_WRITE)
                mm.write(actions.tobytes())
                mm.close()
        else:
            total_actions = 0

        # Pass in fire prediction settings
        params = f"{int(preprocess_actions)} {batch_size} {total_actions} {self.bias} {self.time_horizon_hr} {self.time_step} {self.cell_size} {self.shape[0]} {self.shape[1]} {self.wind_t_step} {self.wind_forecast_str}"

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        executable_path = os.path.join(current_script_dir, '..', 'executable', 'fire_prediction')

        if total_actions > 0:
            result = subprocess.run([executable_path, self.cell_filename, self.action_filename], input=params, text=True, capture_output=True)
        else:
            result = subprocess.run([executable_path, self.cell_filename], input=params, text=True, capture_output=True)

        if result.returncode != 0:
            print("Executable terminated with the following errors: ")
            print("Error output: ", result.stderr)

        predictions = self.read_data(self.output_path)

        return predictions

    def run_prediction(self, action_sequence:list = None, preprocess_actions:bool = False) -> dict:
        """Run a prediction

        :param action_sequence: Action sequence that should be completed during the course of 
                                the prediction run. Specify as a list of
                                :class:`~utilities.action.Action` objects.
        :return: Dictionary where each key is a time-step and each value is a list of predicted
                 ignition locations (x, y) at that time-step. Time-steps start at the time-step the
                 original fire when input to the prediction model.
        :rtype: dict
        """
        batch_size = 1

        if action_sequence is not None:
            action_vec = [action_sequence]

        else:
            action_vec = None

        pred_vec = self.run_prediction_batch(batch_size, action_vec, preprocess_actions)

        return pred_vec[0]

    def convert_to_action_type(self, batch_action_sequence):

        total_actions = sum(len(seq) for seq in batch_action_sequence)

        actions = np.zeros(total_actions + len(batch_action_sequence), dtype=action_type)
        idx = 0

        for action_seq in batch_action_sequence:
            actions[idx]['time'] = len(action_seq)
            idx += 1
            for action in action_seq:
                actions[idx]['time'] = action.time
                actions[idx]['pos']['x'] = action.loc[0]
                actions[idx]['pos']['y'] = action.loc[1]

                if isinstance(action, SetFuelMoisture):
                    actions[idx]['type'] = 0
                    actions[idx]['value'] = action.moisture
                
                elif isinstance(action, SetFuelContent):
                    actions[idx]['type'] = 1
                    actions[idx]['value'] = action.content

                elif isinstance(action, SetIgnition):
                    fire_type = action.fire_type
                    if fire_type == FireTypes.PRESCRIBED:
                        actions[idx]['type'] = 2
                    
                    else:
                        actions[idx]['type'] = 3

                idx += 1
        
        return actions, total_actions

    def read_data(self, filename):
        # Open the file and memory-map it
        with open(filename, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Start reading from the beginning
            offset = 0
            num_maps = struct.unpack_from('i', mm, offset)[0]

            offset += struct.calcsize('i')

            all_predictions = []

            for _ in range(num_maps):
                # Read an integer for the number of time steps in this map
                num_entries = struct.unpack_from('i', mm, offset)[0]

                offset += struct.calcsize('i')

                prediction_map = {}

                for __ in range(num_entries):
                    # Read a float for the time step
                    time_step = struct.unpack_from('f', mm, offset)[0]
                    offset += struct.calcsize('f')

                    # Read an integer for the number of cells
                    num_cells = struct.unpack_from('i', mm, offset)[0]
                    offset += struct.calcsize('i')

                    prediction_map[time_step] = []

                    # Loop through the number of cells
                    for ___ in range(num_cells):
                        # Each cell consists of two floats (x, y)
                        x, y = struct.unpack_from('ff', mm, offset)
                        offset += struct.calcsize('ff')

                        prediction_map[time_step].append((x, y))

                all_predictions.append(prediction_map)

            # Close the mmap
            mm.close()

        # Optionally remove the file after reading
        os.remove(filename)

        return all_predictions

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