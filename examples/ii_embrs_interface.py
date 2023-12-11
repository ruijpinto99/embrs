"""Demo class demonstrating some useful functions of the fire's interface.

To run this example code, start a fire sim and select this file as the "User Module"
"""

from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from embrs.utilities.fire_util import CellStates

import numpy as np

class InterfaceDemo(ControlClass):
	def __init__(self, fire:FireSim):

		# Get map size in meters
		width_m, height_m = fire.size
		print(f'Map is {width_m} m x {height_m} m')

		# Save x and y coordinate for later
		self.x = int(np.floor(width_m/4))
		self.y = int(np.floor(height_m/4))

		# Get shape of map's backing array
		rows, cols = fire.shape
		print(f'Map backing array has {rows} rows and {cols} cols')

		# Save the middle row and col for later
		self.row = int(np.floor(rows/2))
		self.col = int(np.floor(cols/2))

		# Get simulation time step
		self.time_step_s = fire.time_step
		print(f'Sim time step is {self.time_step_s}')

		# Variable to keep track of wind updates
		self.last_wind_update = 0

	def process_state(self, fire:FireSim):
		# Check the current sim time and perform cell operations at 30 minutes
		if np.abs((fire.curr_time_m - 30)) == 0:
			print("Performing cell operations")

			# Get the cell containing the saved coordinates
			cell_from_x_y = fire.get_cell_from_xy(self.x, self.y)

			# Get the state of the cell
			cell_state = cell_from_x_y.state

			# Set the fuel content and fuel moisture of the cell if it's unburnt 
			if cell_state == CellStates.FUEL:
				fuel_content = 0.25
				fire.set_fuel_content_at_cell(cell_from_x_y, fuel_content)

				# Get the dead moisture of extinction for the fuel at the cell
				fuel_moisture =  cell_from_x_y.fuel_type.dead_m_ext
				
				# Set the fuel moisture to half the moisture of extinction
				fire.set_fuel_moisture_at_cell(cell_from_x_y, fuel_moisture)

			# Get the cell at the saved indices
			cell_from_row_col = fire.get_cell_from_indices(self.row, self.col)

			# Get the state of the cell
			cell_state = cell_from_row_col.state

			# Set a prescribed burn at cell if it's unburnt
			if cell_state == CellStates.FUEL:
				fire.set_prescribed_fire_at_cell(cell_from_row_col)

		# Check if a wind time step has passed and print out the updated speed and direction
		if fire.curr_time_m - self.last_wind_update > fire.wind_vec.time_step:
			print("Wind updated")
			
			# Get the new wind conditions
			speed_m_s, dir_deg = fire.get_curr_wind_speed_dir()

			print(f"Wind speed: {speed_m_s} m/s")
			print(f"Wind direction: {dir_deg} deg")

			self.last_wind_update = fire.curr_time_m

		# Print the average fire coordinate every 100 iterations of the sim
		if fire.iters % 100 == 0:
			x_avg, y_avg = fire.get_avg_fire_coord()
			
			print(f"Avg. Fire Coordinate: ({x_avg} m, {y_avg} m)")