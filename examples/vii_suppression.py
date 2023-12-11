"""Sample class which carries out suppression operations based on the location of 
the frontier of the fire.

Note: This example is not intended to be a robust firefighting algorithm, it is to be used 
as a reference on how agents can be used in conjuction with the fire interface to formulate
a response to a fire.

To run this example code, start a fire sim and select this fiel as the "User Module"
"""

import numpy as np
from sklearn.cluster import KMeans

from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from v_sample_agent import SampleAgent

class SuppressionExample(ControlClass):
    def __init__(self, fire:FireSim):
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
            a.max_tasks = np.inf

            fire.add_agent(a)
            self.agents.append(a)

        self.soak_complete = False

    # process_state is called by the FireSim every iteration
    def process_state(self, fire: FireSim) -> None:
        # Check if it is time to dispatch agents
        if fire.curr_time_h > self.operation_start_time_h:
            if all(len(a.task_list) == 0 for a in self.agents):
                # Create tasks
                clusters, means = self.create_soak_tasks(fire)
                
                # Assign tasks
                self.assign_soak_tasks(clusters, means)

                self.soak_complete = True

        # Step each agent forward one time step
        self.iterate_agents()

    def create_soak_tasks(self, fire: FireSim) -> list:
        tot_x = 0
        tot_y = 0

        # Get the fire frontier sum up x and y coordinates
        frontier_points = []
        for cell in fire.frontier:
            frontier_points.append((cell.x_pos, cell.y_pos))
            tot_x += cell.x_pos
            tot_y += cell.y_pos

        # Calculate the average fire frontier x and y coordinates
        avg_frontier_x = tot_x / len(frontier_points)
        avg_frontier_y = tot_y / len(frontier_points)

        # Generate vectors from the average frontier coordinate to each frontier point
        frontier_vecs = [[p[0] - avg_frontier_x, p[1] - avg_frontier_y] for p in frontier_points]

        # Generate task points by projecting out 300 meters from each frontier point
        task_points = []
        for i, p in enumerate(frontier_points):
            unit_proj_vec = frontier_vecs[i]/np.linalg.norm(frontier_vecs[i])
            task_points.append([p[0] + 300 * unit_proj_vec[0], p[1] + 300 * unit_proj_vec[1]])

        # group task points into a group for each agent
        num_groups = np.min([len(self.agents), len(task_points)])

        # Use kmeans to cluster the task points
        kmeans = KMeans(n_clusters=num_groups, n_init=10)
        kmeans.fit(task_points)

        clusters = {}
        for i, label in enumerate(kmeans.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(task_points[i])

        means = kmeans.cluster_centers_.tolist()

        return clusters, means

    def assign_soak_tasks(self, clusters, means):
        # A boolean list to track assigned clusters
        assigned_tasks = [False] * len(clusters)
        
        # Iterate through agents and calculate the best cluster for each agent based on distance
        for agent in self.agents:
            min_dist = np.inf
            best_cluster = None

            for i, mean in enumerate(means):
                if not assigned_tasks[i]:  # Check if this cluster has been assigned
                    dist = np.sqrt((agent.x - mean[0])**2 + (agent.y - mean[1])**2)

                    if dist < min_dist:
                        min_dist = dist  # Update min_dist
                        best_cluster = i

            task_list = clusters[best_cluster]

            # Append tasks to selected agent's task list
            for task in task_list:
                agent.task_list.append((task, 'soak'))

            assigned_tasks[best_cluster] = True  # Mark this cluster as assigned

    # Call the complete_tasks function for each agent each time-step
    def iterate_agents(self):
        for agent in self.agents:
            agent.complete_tasks()




