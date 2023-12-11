"""Demo class demonstrating the use of custom logging during a fire simulation.
These log messages will appear in the "messages" section of the status_log.json file

To run this example code, start a fire sim and select this file as the "User Module"
"""

from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim


class CustomLogging(ControlClass):
	def __init__(self, fire:FireSim):
		# Log custom messages with fire.logger.log_message()
		fire.logger.log_message("Custom Logging Code Running")

		self.operation_performed = False
		self.area_messaged_logged = False

	# process_state is called by the FireSim every iteration
	def process_state(self, fire: FireSim) -> None:
		# Log a message when some operation is performed
		if fire.curr_time_h > 1 and not self.operation_performed:
			self.perform_some_operation(fire)
		
		# Calculate burning area of the fire
		burning_area = len(fire.curr_fires) * fire.cell_grid[0, 0].cell_area

		# Log a message when burning area exceeds a certain value
		if burning_area > 5000 and not self.area_messaged_logged:
			fire.logger.log_message(f"Burning area exceeded 5000 m^2")
			self.area_messaged_logged = True
		
	def perform_some_operation(self, fire):
		fire.logger.log_message(f"Some operation performed at {fire.curr_time_h} hours")
		self.operation_performed = True