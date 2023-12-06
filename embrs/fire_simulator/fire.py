"""Core fire simulation model.

.. autoclass:: FireSim
    :members:
"""

from tqdm import tqdm
import numpy as np

from embrs.base_classes.base_fire import BaseFireSim
from embrs.utilities.fire_util import CellStates, FireTypes
from embrs.utilities.fire_util import ControlledBurnParams
from embrs.fire_simulator.cell import Cell
from embrs.fire_simulator.wind import Wind

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
    def __init__(self, fuel_map: np.ndarray, fuel_res: float, topography_map: np.ndarray,
                topography_res: float, wind_vec: Wind, roads: list, fire_breaks: list,
                time_step: int, cell_size: float, duration_s: float, initial_ignition: list,
                size: tuple, display_freq_s = 300):
        """Constructor method to initialize a fire simulation instance. Saves input parameters,
        creates backing array, populates cell_grid and cell_dict with cells, sets initial ignition,
        applies fire-breaks and roads.
        """

        print("Simulation Initializing...")

        # Fuel fraction to be considered BURNT
        self.burnout_thresh = 0.1

        self.logger = None
        self.progress_bar = None

        self._updated_cells = {}
        self._curr_updates = []

        self._partially_burnt = []
        self._soaked = []

        self._agent_list = []
        self._agents_added = False

        self._curr_fires_cache = []
        self._curr_fires_anti_cache = []
        self._curr_fires = set()

        self.w_vals = []
        self.fuels_at_ignition = []
        self.ignition_clocks = []

        super().__init__(fuel_map, fuel_res, topography_map,
                topography_res, wind_vec, roads, fire_breaks,
                time_step, cell_size, duration_s, initial_ignition,
                size, display_freq_s= display_freq_s)

    def iterate(self):
        """Step forward the fire simulation a single time-step
        """
        # Set-up iteration
        if self._init_iteration():
            return

        # Update wind
        self.wind_changed = self._wind_vec._update_wind(self._curr_time_s)

        # Iterate over currently burning cells
        for curr_cell in self._curr_fires:

            # iterate over burnable neighbors
            neighbors_to_remove = []
            for neighbor_id, (dx, dy) in curr_cell._burnable_neighbors:
                neighbor = self._cell_dict[neighbor_id]

                neighbor_burnable = self.check_if_burnable(curr_cell, neighbor)

                if not neighbor_burnable:
                    if neighbor.fire_type != FireTypes.PRESCRIBED:
                        neighbors_to_remove.append((neighbor_id, (dx, dy)))

                else:
                    prob, v_prop = self._calc_prob(curr_cell, neighbor, (dx, dy))

                    if np.random.random() < prob:
                        neighbor._set_vprop(v_prop)
                        self.set_state_at_cell(neighbor, CellStates.FIRE, curr_cell.fire_type)
                        # Add cell to update dictionary
                        self._updated_cells[neighbor.id] = neighbor

                        if neighbor.to_log_format() not in self._curr_updates:
                            self._curr_updates.append(neighbor.to_log_format())


                        if neighbor in self._frontier:
                            self._frontier.remove(neighbor)
                    else:
                        if neighbor not in self._frontier:
                            self._frontier.add(neighbor)

            curr_cell._burnable_neighbors.difference_update(neighbors_to_remove)

            self.capture_cell_changes(curr_cell)

        self.update_fuel_contents()

        self.log_changes()

    def log_changes(self):
        """Log the changes in state from the current iteration
        """
        self._curr_updates.extend(self._partially_burnt)
        self._curr_updates.extend(self._soaked)
        self._soaked = []
        self._iters += 1
        self.logger.add_to_cache(self._curr_updates.copy(), self.curr_time_s)

        if self.agents_added:
            self.logger.add_to_agent_cache(self._get_agent_updates(), self.curr_time_s)

    def update_fuel_contents(self):
        """Update the fuel content of all the burning cells based on the mass-loss algorithm in 
            `(Coen 2005) <http://dx.doi.org/10.1071/WF03043>`_
        """
        curr_fires = self._curr_fires
        burnout_thresh = self.burnout_thresh
        time_step = self.time_step

        ignition_clocks = np.array(self.ignition_clocks)
        fuels_at_ignition = np.array(self.fuels_at_ignition)
        w_vals = np.array(self.w_vals)

        ignition_clocks += time_step
        fuel_contents = fuels_at_ignition * np.minimum(1, np.exp(-ignition_clocks/w_vals))

        relational_dict = self._relational_dict
        updated_cells = self._updated_cells
        curr_updates = self._curr_updates

        for cell, fuel_content, ignition_clock in zip(curr_fires, fuel_contents, ignition_clocks):
            cell._fuel_content = fuel_content
            cell.ignition_clock = ignition_clock

            burnout_thresh = cell.fuel_at_ignition * ControlledBurnParams.burnout_fuel_frac if cell.fire_type == FireTypes.PRESCRIBED else self.burnout_thresh

            if fuel_content < burnout_thresh:
                self.set_state_at_cell(cell, CellStates.BURNT)
                # Add cell to update dictionary
                self._updated_cells[cell.id] = cell

                if cell.to_log_format() not in self._curr_updates:
                    self._curr_updates.append(cell.to_log_format())


                for neighbor_id, _ in cell.neighbors:
                    key = (cell.id, neighbor_id)
                    if key in relational_dict:
                        del relational_dict[key]

            updated_cells[cell.id] = cell
            curr_updates.append(cell.to_log_format())

    def capture_cell_changes(self, curr_cell: Cell):
        """Update data structures with values relating to a burning cell's fuel.
        
        Data structures are later used by :py:attr:`~fire_simulator.fire.update_fuel_contents`

        :param curr_cell: Burning cell to grab values from
        :type curr_cell: Cell
        """
        self.w_vals.append(curr_cell.W)
        self.fuels_at_ignition.append(curr_cell.fuel_at_ignition)
        self.ignition_clocks.append(curr_cell.ignition_clock)

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

        if len(self._curr_fires) == 0 or self._curr_time_s >= self._sim_duration:
            self._finished = True
            self.progress_bar.close()

            if len(self._curr_fires) == 0:
                self.logger.log_message("Fire extinguished! Terminating early.")
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
