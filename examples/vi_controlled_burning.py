"""Sample class which carries out prescribed burns based on the current wind conditions
and the locations of the fire breaks and roads.

Note: This example is not intended to be a robust firefighting algorithm, it is to be used
as a reference on how agents can be used in conjuction with the fire interface to formulate
a response to a fire. It does not, for example, handle the case where there is no wind.

This example works well when using the provided example map titled 'burnout_map' and the sample
wind forecast titled 'burnout_wind_forecast'

To run this example code, start a fire sim and select this file as the "User Module"
"""

import heapq
import numpy as np

from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from v_sample_agent import SampleAgent

class BurnoutExample(ControlClass):
	def __init__(self, fire: FireSim):
		# Define time agents should dispatch
		self.operation_start_time_h = 0.33

		# Initialize empty list to store agents
		self.agents = []

		# Define number of agents
		num_agents = 10

		# Randomly place agents
		for i in range(num_agents):
			a_x = np.random.random() * fire.x_lim
			a_y = np.random.random() * fire.y_lim

			a = SampleAgent(i, a_x, a_y, fire, label=f"{i}")
			a.max_tasks = 10

			fire.add_agent(a)
			self.agents.append(a)

		self.burnout_complete = False

	# process_state is called by the FireSim every iteration
	def process_state(self, fire: FireSim) -> None:
		# Check if it is time to dispatch agents
		if fire.curr_time_h > self.operation_start_time_h:
			if not self.burnout_complete:
				# Create tasks
				burn_tasks = self.create_burn_tasks(fire)
				
				# Assign tasks
				self.assign_burn_tasks(burn_tasks, 50, fire)

				self.burnout_complete = True

		# Step each agent forward one time step
		self.iterate_agents()

	def create_burn_tasks(self, fire: FireSim) -> list:
		# Get the average fire coordinate
		avg_fire_x, avg_fire_y = fire.get_avg_fire_coord()

		# Retrieve current wind conditions
		wind_vec = fire.get_curr_wind_vec()
		wind_mag = np.linalg.norm(wind_vec	)
		wind_unit_vec = wind_vec/wind_mag

		# Retrieve locations of fire breaks
		fire_break_points = []
		for cell in fire.fire_break_cells:	
			fire_break_points.append((cell.x_pos, cell.y_pos))

		# Retrieve locations of roads
		if fire.roads is not None:
			for road in fire.roads:
				for point in road[0]:
					road_x, road_y = point[0], point[1]
					fire_break_points.append((road_x, road_y))

		# Create vectors from avg fire position to the fire break and road points
		fire_break_vectors = [[p[0] - avg_fire_x, p[1] - avg_fire_y] for p in fire_break_points]

		# Calculate angles between fire break vectors and wind
		angles = [vector_angle(wind_unit_vec, fbv) for fbv in fire_break_vectors]
		
		# Filter out angles that are outside of 90 degrees from the wind direction
		within_cone = [fire_break_points[i] for i, angle in enumerate(angles) if angle <= 90]

		# Create vector that points in oppostive direction of wind
		projection_vec = -500*np.array(wind_unit_vec)

		# Project points from fire breaks using the above vector, these are used as task points
		task_points = [[p[0] + projection_vec[0], p[1] + projection_vec[1]] for p in within_cone]

		return task_points
	
	def assign_burn_tasks(self, tasks, spacing, fire):
		tasks_copy = tasks.copy()

		# List to keep track of all assigned tasks
		assigned_tasks = []  

		# Get average fire coordinate
		fire_x, fire_y = fire.get_avg_fire_coord()

		# Create empty list of task penalties
		tasks_penalties = np.empty(len(tasks_copy))

		# Calculate task penalties for each task based on distance from avg fire coordinate
		for i, t in enumerate(tasks_copy):
			t_dist = np.sqrt((t[0] - fire_x)**2 + (t[1] - fire_y)**2)
			tasks_penalties[i] = t_dist

		# Assign each agent with tasks
		for agent in self.agents:
			# Calculate distances from agent to all tasks
			distances = [np.sqrt((agent.x - task[0])**2 +
						(agent.y - task[1])**2) for task in tasks_copy]

			# Create a min-heap of distances and indices
			k = 2
			heap = [(dist+(k*tasks_penalties[i]), i) for i, dist in enumerate(distances)]
			heapq.heapify(heap)

			while len(agent.task_list) < agent.max_tasks and heap:
				# Get the task with the smallest distance
				_, best_task_index = heapq.heappop(heap)
				# Check if this task is > 50m away from all tasks already assigned to any agent
				best_task = tasks_copy[best_task_index]
				if all(np.sqrt((t[0] - best_task[0])**2 + (t[1] - best_task[1])**2) > spacing
						for t in assigned_tasks):

					# Add best task to the agent's task list
					agent.task_list.append((best_task, 'burn'))
					# Also add it to the list of all assigned tasks
					assigned_tasks.append(best_task)

	# Call the complete_tasks function for each agent each time-step
	def iterate_agents(self):
		for agent in self.agents:
			agent.complete_tasks()

# Function to calculate angle between vectors
def vector_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    mags = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(dot_product/mags))
