"""Core fire simulation model.

.. autoclass:: FireSim
    :members:
"""

from tqdm import tqdm
import numpy as np
from typing import Tuple

from embrs.base_classes.base_fire import BaseFireSim
from embrs.utilities.fire_util import CellStates, FireTypes, FuelConstants, UtilFuncs, HexGridMath
from embrs.utilities.fire_util import ControlledBurnParams
from embrs.fire_simulator.cell import Cell
from embrs.fire_simulator.wind import Wind

from embrs.utilities.rothermel import *

class FireSim(BaseFireSim):
    """Fire simulator class utilizing a probabilistic fire model.
    
    Based on the work of `Trucchia et. al <https://www.mdpi.com/2571-6255/3/3/26>`_. The basis of
    the simulator is a hexagonal grid, each cell in the  grid is a 
    :class:`~fire_simulator.cell.Cell` object that contains all the pertinent fire spread parameters
    for that cell. The simulator takes into account the effects of fuel, slope, and wind on the
    fire propagation.

    :param fuel_map: 2D array that represents the spatial distribution of fuel types in the sim.
                     Each element must be one of the
                     `13 Anderson FBFMs <https://www.fs.usda.gov/rm/pubs_int/int_gtr122.pdf>`_.
                     Note: array is flipped across the horizontal axis after input.
    :type fuel_map: np.ndarray
    :param fuel_res: Resolution of the fuel map in meters. How many meters each row/column
                     represents in space
    :type fuel_res: float
    :param topography_map: 2D array that represents the elevation in meters at each point in space.
                           Note: array is flipped across the
                           horizontal axis after input.
    :type topography_map: np.ndarray
    :param topography_res: Resolution of the topography map in meters. How many meters each
                           row/column represents in space.
    :type topography_res: float
    :param wind_vec: Wind object defining the wind conditions for the simulation.
    :type wind_vec: :class:`~fire_simulator.wind.Wind`
    :param roads: List of points that will be considered roads for the simulation. Format for each
                  element in list: ((x,y), fuel_content). Where (x,y) is a tuple containing the
                  spatial in the sim measured in meters, and fuel_content is the amount of fuel to
                  be modeled at that point (between 0 and 1)
    :type roads: list
    :param fire_breaks: List of dictionaries representing fire-breaks where each dictionary has a
                        "geometry" key with a :py:attr:`shapely.LineString` value and a 
                        "fuel_value" key with a float value which represents the amount of fuel
                        modeled along the :py:attr:`shapely.LineString`.
    :type fire_breaks: list
    :param time_step: Time step of the simulation, defines the amount of time each iteration will
                      model.
    :type time_step: int
    :param cell_size: Size of each cell in the simulation, measured as the distance in meters
                    between two parallel sides of the regular hexagon cells.
    :type cell_size: float
    :param duration_s: Duration of time (in seconds) the simulation should run for, the sim will
                       run for this duration unless the fire is extinguished before the duration
                       has passed.
    :type duration_s: float
    :param initial_ignition: List of shapely polygons that represent the regions of the sim that
                             should start as initially on fire.
    :type initial_ignition: list
    :param size: Size of the simulation backing array (rows, cols)
    :type size: tuple
    :param display_freq_s: The amount of time (in seconds) between updating the real-time
                           visualizer, only used if real-time visualization is selected, defaults
                           to 300
    :type display_freq_s: int, optional
    """
    def __init__(self, fuel_map: np.ndarray, fuel_res: float, topography_map: np.ndarray, topography_res: float, aspect_map: np.ndarray,
                aspect_res: float, slope_map: np.ndarray, slope_res: float, wind_vec: Wind, roads: list, fire_breaks: list,
                time_step: int, cell_size: float, duration_s: float, initial_ignition: list,
                size: tuple, burnt_cells: list = None, display_freq_s = 300):
        """Constructor method to initialize a fire simulation instance. Saves input parameters,
        creates backing array, populates cell_grid and cell_dict with cells, sets initial ignition,
        applies fire-breaks and roads.
        """

        print("Simulation Initializing...")

        # Fuel fraction to be considered BURNT
        self.burnout_thresh = FuelConstants.burnout_thresh

        self.logger = None
        self.progress_bar = None

        self._updated_cells = {}
        self._curr_updates = []

        self._partially_burnt = []
        self._soaked = []

        self._agent_list = []
        self._agents_added = False

        self._ignition_schedule = {}

        self.fireline_ign_threshold = 0.0 #kW/m # TODO: set this properly


        super().__init__(fuel_map, fuel_res, topography_map, topography_res, aspect_map,
                aspect_res, slope_map, slope_res, wind_vec, roads, fire_breaks,
                time_step, cell_size, duration_s, initial_ignition,
                size, burnt_cells = burnt_cells, display_freq_s= display_freq_s)

        self._init_iteration()

    def iterate(self):
        """Step forward the fire simulation a single time-step
        """
        # Set-up iteration
        if self._init_iteration():
            self._finished = True
            return

        # Update wind
        self.wind_changed = self._wind_vec._update_wind(self._curr_time_s)

        # Check schedule for ignitions at the current time step
        new_ignitions = self.get_scheduled_ignitions()

        if self._iters == 0:
            new_ignitions = self.starting_ignitions

        for cell, loc in new_ignitions:
            # Set cell state to burning
            cell._set_state(CellStates.FIRE)
            cell._set_fire_type(FireTypes.WILD) # TODO: Need to have this set by the cell that ignited its fire type
            self._updated_cells[cell.id] = cell

            if cell.to_log_format() not in self._curr_updates:
                self._curr_updates.append(cell.to_log_format())

            directions, distances, end_points = UtilFuncs.get_ign_parameters(loc, self.cell_size)

            rothermel_data = calc_propagation_in_cell(cell.fuel_type, directions, self._wind_vec.wind_speed, self._wind_vec.wind_dir_deg, cell.slope_deg, cell.aspect)

            for i, (r_gamma, I_gamma) in enumerate(rothermel_data):
                if I_gamma > self.fireline_ign_threshold:
                    self.schedule_ignition(cell, r_gamma, distances[i], end_points[i])

        self.log_changes()

    def schedule_ignition(self, cell: Cell, r_gamma: float, dist: float, end_point):
        # Calculate how long fire will take to reach end point given local ROS
        t_to_end_point = dist / r_gamma

        ign_time = self.curr_time_s + t_to_end_point
        schedule_t = int(np.ceil(ign_time / self._time_step) * self._time_step)

        for pt in end_point:
            n_loc = pt[0]
            neighbor = self.get_neighbor_from_end_point(cell, pt)

            # if r_gamma > 0.1:
            #     print(f"neighbor: {neighbor}")
            #     print(f"schedule_t: {schedule_t} sec")

            if neighbor:
                # print(f"Scheduling ignition in: {schedule_t / 60} mins")

                # Check that neighbor state is burnable
                if neighbor.state == CellStates.FUEL and neighbor.fuel_type.burnable: # TODO: Should we include prescribed burns 
                    
                    if not (neighbor.scheduled and neighbor.scheduled_t <= schedule_t):

                        # print(neighbor.scheduled)
                        # print(f"neighbors scheduled_time: {neighbor.scheduled_t}")
                        # print()


                        # Add ignition to the schedule
                        if self._ignition_schedule.get(schedule_t):
                            self._ignition_schedule[schedule_t].append((neighbor, n_loc))

                        else:
                            self._ignition_schedule[schedule_t] = [(neighbor, n_loc)]
                        
                        neighbor.scheduled = True
                        neighbor.scheduled_t = schedule_t

    def get_scheduled_ignitions(self):
        # Make sure curr time is an int
        curr_time_key = int(self.curr_time_s)

        # Get scheduled ignitions
        new_ignitions = self._ignition_schedule.get(curr_time_key, [])

        # Delete entry from schedule
        if new_ignitions:
            del self._ignition_schedule[curr_time_key]

        return new_ignitions

    def get_neighbor_from_end_point(self, cell, end_point) -> Cell:
        neighbor = None

        neighbor_letter = end_point[1]
        # Get neighbor based on neighbor_letter
        if cell._row % 2 == 0:
            diff_to_letter_map = HexGridMath.even_neighbor_letters
            
        else:
            diff_to_letter_map = HexGridMath.odd_neighbor_letters

        dx, dy = diff_to_letter_map[neighbor_letter]

        row_n = int(cell.row + dy)
        col_n = int(cell.col + dx)

        if self._grid_height >= row_n >=0 and self._grid_width >= col_n >= 0:
            neighbor = self._cell_grid[row_n, col_n]

        return neighbor

    def get_rel_wind_direction(self, slope_dir):
        # TODO: Change wind definition to align with 0 degree north
        rel_wind_dir = self._wind_vec.wind_dir_deg - slope_dir 
        if rel_wind_dir < 0:
            rel_wind_dir += 360

        return rel_wind_dir

    def get_cell_slope(self, cell):
        # TODO: check that this calculation works correctly

        # Estimate the slope effect within a single cell based on the topography around it
        dx_total = 0
        dy_total = 0

        for neighbor_id, _ in cell.neighbors:
            # Get neighbor
            neighbor = self._cell_dict[neighbor_id]

            # Calculate change in elevation
            d_elev = neighbor.z - cell.z

            # Get unit vector pointing in direction of neighbor
            dx = neighbor.x_pos - cell.x_pos
            dy = neighbor.y_pos - cell.y_pos
            dist = np.sqrt(dx ** 2 + dy ** 2)
            dx /= dist
            dy /= dist

            # Add difference in elevation distributed along unit vector
            dx_total += d_elev * dx
            dy_total += d_elev * dy

        # Calculate average slope percentage
        rise = np.sqrt(dx_total**2 + dy_total**2)
        run = dist
        slope_pct = (rise/run) * 100

        # Calculate average slope direction
        slope_dir_rad = np.arctan2(dy_total, dx_total)
        slope_dir_deg = 360 - (np.rad2deg(slope_dir_rad) - 90)

        if slope_dir_deg < 0:
            slope_dir_deg += 360

        # If slope is negative, convert it to a positive value
        if slope_pct < 0:
            slope_pct = -slope_pct
            slope_dir_deg = (slope_dir_deg + 180) % 360

        slope_angle_deg = abs(np.rad2deg(np.arctan(rise/run)))

        return slope_angle_deg, slope_dir_deg

    def log_changes(self):
        """Log the changes in state from the current iteration
        """
        self._curr_updates.extend(self._partially_burnt)
        self._curr_updates.extend(self._soaked)
        self._soaked = []
        self._iters += 1
        if self.logger:
            self.logger.add_to_cache(self._curr_updates.copy(), self.curr_time_s)

            if self.agents_added:
                self.logger.add_to_agent_cache(self._get_agent_updates(), self.curr_time_s)


    def _init_iteration(self) -> bool:
        """Set up the next iteration. Reset and update relevant data structures based on last 
        iteration. 

        :return: Boolean value representing if the simulation should be terminated.
        :rtype: bool
        """
        if self._iters == 0:
            self.progress_bar = tqdm(total=self._sim_duration/self.time_step,
                                     desc='Current sim ', position=0, leave=False)

        self._curr_updates.clear()

        # Update current time
        self._curr_time_s = self.time_step * self._iters
        self.progress_bar.update()

        self._curr_fires = self._curr_fires | set(self._curr_fires_cache)
        self._curr_fires.difference_update(self._curr_fires_anti_cache)

        self._curr_fires_cache = []
        self._curr_fires_anti_cache = []

        self.w_vals = []
        self.fuels_at_ignition = []
        self.ignition_clocks = []

        if self._curr_time_s >= self._sim_duration: # TODO: need a way to check if no more ignitions scheduled
            self.progress_bar.close()

            return True

        return False

    def _get_agent_updates(self):
        """Returns a list of dictionaries describing the location of each agent in __agent_list

        :return: List of dictionaries, describing the x,y position of each agent as well as their
                 display preferences
        :rtype: list
        """
        agent_data = []

        for agent in self.agent_list:
            agent_data.append(agent.to_log_format())

        return agent_data

    @property
    def updated_cells(self) -> dict:
        """Dictionary containing cells updated since last time real-time visualization was updated.
        Dict keys are the ids of the :class:`~fire_simulator.cell.Cell` objects.
        """
        return self._updated_cells

    @property
    def curr_updates(self) -> list:
        """List of cells updated during the most recent iteration.
        
        Cells are in their log format as generated by 
        :func:`~fire_simulator.cell.Cell.to_log_format()`
        """
        return self._curr_updates


    @property
    def agent_list(self) -> list:
        """List of :class:`~base_classes.agent_base.AgentBase` objects representing agents
        registered with the sim.
        """

        return self._agent_list

    @property
    def agents_added(self) -> bool:
        """`True` if agents have been registered in sim's agent list, returns `False` otherwise
        """
        return self._agents_added
