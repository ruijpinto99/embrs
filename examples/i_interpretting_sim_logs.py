"""Demo script showing how to access the state of a sim at any snapshot in time
"""
import pickle
import msgpack
import itertools
from embrs.utilities.fire_util import UtilFuncs
from embrs.utilities.fire_util import CellStates

# Paths to init_fire_state.pkl and pkl log life
INIT_STATE = '/path to log folder/init_fire_state.pkl'
LOG_FILE = '/path to log folder/run_#/log.msgpack'

TIME = 1000 # snapshot in time of interest, make sure this is in range of (0-sim_duration) and that
            # it is an integer multiple of time_step

# Load the initial state
with open(INIT_STATE, 'rb') as f:
    sim = pickle.load(f)

# Retrieve properties from sim dictionary
wind = sim['wind_vec']
cell_dict = sim['cell_dict'] 
cell_grid = sim['cell_grid']

cell_size = sim['cell_size']
grid_width = sim['grid_width']
grid_height = sim['grid_height']

# Load the log file
with open(LOG_FILE, 'rb') as f:
    data = msgpack.unpack(f, strict_map_key=False) # Note: you must use strict_map_key=False when unpacking

time_steps = data.keys() # gives a list of all the time steps that occurred during the simulation

for time in time_steps:
    if time > TIME:
        break

    updated_cells = data[time] # get the cells that were updated during this particular time step

    wind._update_wind(time) # update the wind vector at this time step if that is of interest

    for cell in updated_cells:
        c = cell_dict[cell['id']] # get the cell object

        c._set_state(cell['state'])
        if c.state == CellStates.FIRE:
            c._set_fire_type(cell['fire_type'])
        c._set_fuel_content(cell['fuel_content'])

# cell_dict now contains a dictionary of all the cells at time 't', each key is the id of the cell
# cell_grid contains the cells in grid form at time 't'

# Cells can be accessed using get_indices_from_xy() by providing a position in the xy plane
# or directly by row,col
row, col = UtilFuncs.get_indices_from_xy(3970, 22.5, cell_size, grid_width, grid_height)
cell = cell_grid[row, col]

print(f"cell: {cell}")
print(f"cell size: {cell_size}")
