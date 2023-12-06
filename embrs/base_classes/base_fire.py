"""Base class implementation of fire simulation model.

Builds fire simulation and contains core calculations for propagation model. Contains fire
interface properties and functions.

.. autoclass:: BaseFireSim
    :members:
"""

from typing import Tuple
from shapely.geometry import Point
import numpy as np

from embrs.utilities.fire_util import CellStates, FireTypes
from embrs.utilities.fire_util import RoadConstants as rc
from embrs.utilities.fire_util import HexGridMath as hex
from embrs.utilities.fire_util import ControlledBurnParams
from embrs.fire_simulator.cell import Cell
from embrs.fire_simulator.fuel import Fuel
from embrs.fire_simulator.wind import Wind
from embrs.base_classes.agent_base import AgentBase

class BaseFireSim:
    """Base class implementation of fire simulation.
    
    Based on the work of `Trucchia et. al <https://www.mdpi.com/2571-6255/3/3/26>`_. The basis of
    the simulator is a hexagonal grid, each cell in the grid is a :class:`~fire_simulator.cell.Cell`
    object that contains all the pertinent fire spread parameters for that cell. The simulator
    takes into account the effects of fuel, slope, and wind on the fire propagation.

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
                size: tuple, burnt_cells: list = None, display_freq_s = 300):
        """Constructor method to initialize a fire simulation instance. Saves input parameters,
        creates backing array, populates cell_grid and cell_dict with cells, sets initial ignition,
        applies fire-breaks and roads.
        """

        # Set up basic sim parameters
        self.display_frequency = display_freq_s # seconds of sim time, 10 minutes
        self._cell_dict = {}
        self._curr_time_s = 0
        self._iters = 0

        self.logger = None
        self._updated_cells = {}
        self._soaked = []
        self._agent_list = []

        self._size = size

        num_cols = int(np.floor(size[0]/(np.sqrt(3)*cell_size)))
        num_rows = int(np.floor(size[1]/(1.5*cell_size)))

        self._shape = (num_rows, num_cols)

        self._cell_grid = np.empty(self._shape, dtype=Cell)
        self._grid_width = self._cell_grid.shape[1] - 1
        self._grid_height = self._cell_grid.shape[0] - 1

        self._sim_duration = duration_s
        self._time_step = time_step
        self._cell_size = cell_size
        self._curr_fires = set()
        self._relational_dict = {}
        self._curr_fires_cache = []
        self._curr_fires_anti_cache = []
        self._burnt_cells = set()
        self._frontier = set()
        self._roads = roads
        self._fire_break_cells = []
        self._finished = False
        self.wind_changed = True

        # Save scenario data
        self.coarse_topography = np.empty(self._shape)
        self._fire_breaks = fire_breaks
        self.base_topography = topography_map
        self._topography_map = np.flipud(topography_map)
        self.base_fuel_map = fuel_map
        self._fuel_map = np.flipud(fuel_map)
        self._wind_vec = wind_vec
        self._topography_res = topography_res
        self._fuel_res = fuel_res
        self._initial_ignition = initial_ignition

        # Populate cell_grid with cells
        id = 0
        for i in range(num_cols):
            for j in range(num_rows):
                # Initialize cell object
                new_cell = Cell(id, i, j, cell_size)
                cell_x, cell_y = new_cell.x_pos, new_cell.y_pos

                # Set cell elevation from topography map
                top_col = int(np.floor(cell_x/topography_res))
                top_row = int(np.floor(cell_y/topography_res))
                new_cell._set_elev(self._topography_map[top_row, top_col])
                self.coarse_topography[j, i] = new_cell.z

                # Set cell fuel type from fuel map
                fuel_col = int(np.floor(cell_x/fuel_res)) - 1
                fuel_row = int(np.floor(cell_y/fuel_res)) - 1
                fuel_key = self._fuel_map[fuel_row, fuel_col]
                new_cell._set_fuel_type(Fuel(fuel_key))

                # Add cell to the backing array
                self._cell_grid[j,i] = new_cell
                self._cell_dict[id] = new_cell
                id +=1

        # Populate neighbors field for each cell with pointers to each of its neighbors
        self._add_cell_neighbors()

        # Set initial ignitions
        for polygon in initial_ignition:
            minx, miny, maxx, maxy = polygon.bounds

            # Get row and col indices for bounding box
            min_row = int(miny // (cell_size * 1.5))
            max_row = int(maxy // (cell_size * 1.5))
            min_col = int(minx // (cell_size * np.sqrt(3)))
            max_col = int(maxx // (cell_size * np.sqrt(3)))

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    if 0 <= row < size[1] and 0 <= col < size[0]:
                        cell = self._cell_grid[row, col]
                        if polygon.contains(Point(cell.x_pos, cell.y_pos)) and cell.fuel_type.fuel_type <= 13:
                            cell._set_fire_type(FireTypes.WILD)
                            cell._set_state(CellStates.FIRE)
                            self._curr_fires_cache.append(cell)

        if burnt_cells is not None:
            for polygon in burnt_cells:
                minx, miny, maxx, maxy = polygon.bounds

                # Get row and col indices for bounding box
                min_row = int(miny // (cell_size * 1.5))
                max_row = int(maxy // (cell_size * 1.5))
                min_col = int(minx // (cell_size * np.sqrt(3)))
                max_col = int(maxx // (cell_size * np.sqrt(3)))

                for row in range(min_row, max_row + 1):
                    for col in range(min_col, max_col + 1):
                        if 0 <= row < size[1] and 0 <= col < size[0]:
                            cell = self._cell_grid[row, col]
                            if polygon.contains(Point(cell.x_pos, cell.y_pos)):
                                cell._set_state(CellStates.BURNT)
                                self._burnt_cells.add(cell)

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
                    cell._set_fuel_content(fuel_val/100)
                    if cell not in self._fire_break_cells:
                        self._fire_break_cells.append(cell)

        if roads is not None:
            for road in roads:
                for point in road[0]:
                    road_x, road_y = point[0], point[1]

                    road_cell = self.get_cell_from_xy(road_x, road_y, oob_ok = True)

                    if road_cell is not None:
                        if road_cell.state == CellStates.FIRE:
                            road_cell._set_state(CellStates.FUEL)
                        road_cell._set_fuel_content(rc.road_fuel_vals[road[1]])

        print("Initialization complete...")

    def _add_cell_neighbors(self):
        """Populate the "neighbors" property of each cell in the simulation with each cell's
        neighbors
        """
        for j in range(self._shape[1]):
            for i in range(self._shape[0]):
                cell = self._cell_grid[i][j]

                neighbors = []
                if cell.row % 2 == 0:
                    neighborhood = hex.even_neighborhood
                else:
                    neighborhood = hex.odd_neighborhood

                for dx, dy in neighborhood:
                    row_n = int(cell.row + dy)
                    col_n = int(cell.col + dx)

                    if self._grid_height > row_n >= 0 and self._grid_width > col_n >= 0:
                        neighbor_id = self._cell_grid[row_n, col_n].id
                        neighbors.append((neighbor_id, (dx, dy)))

                cell._neighbors = neighbors
                cell._burnable_neighbors = set(neighbors)

    def _calc_prob(self, curr_cell: Cell, neighbor: Cell, disp: tuple) -> Tuple[float, float]:
        """Calculate the probability of curr_cell igniting neighbor at the current instant and the
        propagation velocity of the fire across the cells

        :param curr_cell: Cell that is currently on fire
        :type curr_cell: Cell
        :param neighbor: Neighboring cell to 'curr_cell'
        :type neighbor: Cell
        :param disp: Displacement from curr_cell to neighbor in form of (dx, dy) where
                    dx represents the displacement in columns and dy in rows.
        :type disp: tuple
        :return: Tuple containing the calculated probability of the current cell igniting its
                    neighbor and the propagation velocity of the fire across the cells
                    (prob, v_prop)
        :rtype: Tuple[float, float]
        """
        # Calculate the probability the current cell igniting its neighbor
        # Calculation method pulled from Trucchia et. al (doi:10.3390/fire3030026)
        curr_key = (curr_cell.id, neighbor.id)
        curr_entry = self._relational_dict.get(curr_key)

        # Check if values are cached and neither the wind nor the cells have changed
        if curr_entry and not (self.wind_changed or neighbor.changed or curr_cell.changed):
            return self._relational_dict[curr_key]['prob'], self._relational_dict[curr_key]['v_prop']

        if not curr_entry:
            # Create a dictionary entry if first function call for pairing
            self._relational_dict[curr_key] = {}
            curr_entry = self._relational_dict[curr_key]

            alpha_h, k_phi = self._calc_slope_effect(curr_cell, neighbor)

            curr_entry['alpha_h'] = alpha_h
            curr_entry['k_phi'] = k_phi

        if self.wind_changed or 'alpha_w' not in curr_entry:
            # Update the wind factors if the wind has changed
            alpha_w, k_w = self.wind_vec._calc_wind_effect(curr_cell, disp)

            curr_entry['alpha_w'] = alpha_w
            curr_entry['k_w'] = k_w

        else:
            alpha_w = self._relational_dict[curr_key]['alpha_w']
            k_w = self._relational_dict[curr_key]['k_w']

        if neighbor.changed or 'e_m' not in curr_entry:
            # Update neighbor effects if neighbor state has changed
            dead_m_ext = neighbor.fuel_type.dead_m_ext
            dead_m = neighbor.dead_m
            fm_ratio = dead_m/dead_m_ext
            e_m = self._calc_fuel_moisture_effect(fm_ratio)
            nc_factor = neighbor.fuel_content

            curr_entry['e_m'] = e_m
            curr_entry['nc'] = nc_factor

            neighbor.changed = False

        else:
            e_m = self._relational_dict[curr_key]['e_m']
            nc_factor = self._relational_dict[curr_key]['nc']

        if curr_cell.changed or 'p_n' not in curr_entry:
            # Update the curr_cell effects if its state has changed
            p_n = curr_cell.fuel_type.spread_probs[neighbor.fuel_type.fuel_type]
            v_n = curr_cell.fuel_type.nominal_vel / 60

            if curr_cell.fire_type == FireTypes.PRESCRIBED:
                p_n *= ControlledBurnParams.nominal_prob_adj
                v_n *= ControlledBurnParams.nominal_vel_adj

            curr_entry['p_n'] = p_n
            curr_entry['v_n'] = v_n

            curr_cell.changed = False

        else:
            v_n = self._relational_dict[curr_key]['v_n']
            p_n = self._relational_dict[curr_key]['p_n']

        # Calculate the probability and propagation velocity
        alpha_wh = alpha_w * curr_entry['alpha_h']
        v_prop = v_n * k_w * curr_entry['k_phi']

        if v_prop == 0:
            delta_t_sec = np.inf

        else:
            delta_t_sec = (self._cell_size * 1.5) / v_prop

        num_iters = delta_t_sec/self.time_step

        prob = (1-(1-p_n)**alpha_wh)*e_m
        prob = 1 - (1-prob)**(1/num_iters)
        prob *= nc_factor

        curr_entry['prob'] = prob
        curr_entry['v_prop'] = v_prop

        return prob, v_prop

    def _calc_fuel_moisture_effect(self, fm_ratio: float) -> float:
        """Calculate the fuel moisture effect factor (e_m) based on the fuel moisture ratio

        :param fm_ratio: fuel moisture ratio, defined as dead fuel moisture over the dead moisture
                            of extinction. min(fm_ratio, 1) is used in the calculation of e_m.
        :type fm_ratio: float
        :return: fuel moisture effect factor (e_m)
        :rtype: float
        """

        fm_ratio = min([fm_ratio, 1])
        e_m = -4.5*((fm_ratio-0.5)**3) + 0.5625

        return e_m

    def _calc_slope_effect(self, curr_cell: Cell, neighbor: Cell) -> Tuple[float, float, int]:
        """Calculate the effect of the slope on the spread probability. Returns the slope effect
        factor (alpha_h), the slope angle (rads) between the cells, and an int representing the
        sign of the angle: -1, 0, 1 representing downslope, flat, and upslope respectively.

        :param curr_cell: Cell that is currently on fire.
        :type curr_cell: Cell
        :param neighbor: Neighbor to 'curr_cell' that could potentially be ignited.
        :type neighbor: Cell
        :return: Returns the slope effect factor (alpha_h), the slope angle (rads) between the
                    cells, and an int representing the sign of the angle: -1, 0, 1 representing
                    downslope, flat, and upslope respectively.
        :rtype: Tuple[float, float, int]
        """
        rise = neighbor.z - curr_cell.z
        run = np.sqrt((curr_cell.x_pos - neighbor.x_pos)**2 +
                        (curr_cell.y_pos - neighbor.y_pos)**2)

        slope_pct = (rise/run) * 100

        if slope_pct == 0:
            alpha_h = 1
        elif slope_pct < 0:
            alpha_h = 0.5/(1+np.exp(-0.2*(slope_pct + 40))) + 0.5
        else:
            alpha_h = 1/(1+np.exp(-0.2*(slope_pct - 40))) + 1

        phi = np.arctan(rise/run)

        if phi < 0:
            A = 1
            phi = -phi
        else:
            A = 0

        k_phi = np.exp(((-1)**A)*3.533*np.tan(phi)**1.2)

        return alpha_h, k_phi


    def check_if_burnable(self, curr_cell, neighbor):
        if curr_cell.fire_type == FireTypes.WILD:
            if neighbor.state == CellStates.FUEL or neighbor.fire_type == FireTypes.PRESCRIBED:
                # Check to make sure that neighbor is a combustible type
                if neighbor.fuel_type.fuel_type <= 13:
                    return True

        elif curr_cell.fire_type == FireTypes.PRESCRIBED:
            if neighbor.state == CellStates.FUEL and neighbor.fuel_content > ControlledBurnParams.min_burnable_fuel_content:
                if neighbor.fuel_type.fuel_type <= 13:
                    return True

        return False

    @property
    def frontier(self) -> list:
        """List of cells on the frontier of the fire.
        
        Cells that are in the :py:attr:`CellStates.FUEL` state and neighboring at least one 
        cell in the :py:attr:`CellStates.FIRE` state. Excludes any cells surrounded completely by
        :py:attr:`CellStates.FIRE`.
        """

        front = []
        frontier_copy = set(self._frontier)

        for c in frontier_copy:
            remove = True
            for neighbor_id, _ in c.neighbors:
                neighbor = self.cell_dict[neighbor_id]
                if neighbor.state == CellStates.FUEL:
                    remove = False
                    break

            if remove:
                self._frontier.remove(c)
            else:
                front.append(c)

        return front

    def get_avg_fire_coord(self) -> Tuple[float, float]:
        """Get the average position of all the cells on fire.

        If there is more than one independent fire this will include the points from both.

        :return: average position of all the cells on fire in the form (x_avg, y_avg)
        :rtype: Tuple[float, float]
        """

        x_coords = np.array([cell.x_pos for cell in self.curr_fires])
        y_coords = np.array([cell.y_pos for cell in self.curr_fires])

        return np.mean(x_coords), np.mean(y_coords)


    def get_cell_from_xy(self, x_m: float, y_m: float, oob_ok = False) -> Cell:
        """Returns the cell in the sim that contains the point (x_m, y_m) in the cartesian
        plane.
        
        (0,0) is considered the lower left corner of the sim window, x increases to the
        right, y increases up.

        :param x_m: x position of the desired point in units of meters
        :type x_m: float
        :param y_m: y position of the desired point in units of meters
        :type y_m: float
        :param oob_ok: whether out of bounds input is ok, if set to `True` out of bounds input
                       will return None, defaults to `False`
        :type oob_ok: bool, optional
        :raises ValueError: oob_ok is `False` and (x_m, y_m) is out of the sim bounds
        :return: :class:`~fire_simulator.cell.Cell` at the requested point, returns `None` if the
                 point is out of bounds and oob_ok is `True`
        :rtype: :class:`~fire_simulator.cell.Cell`
        """

        point = Point(x_m, y_m)

        try:
            # Initial estimate of the cell the point might be in
            row = int(y_m // (self._cell_size * 1.5))
            if row % 2 == 0:
                col = int(x_m // (self._cell_size * np.sqrt(3))) + 1
            else:
                col = int((x_m // (self._cell_size * np.sqrt(3))) - 0.5) + 1

            # Check if the estimated cell contains the point
            estimated_cell = self._cell_grid[row, col]

            if estimated_cell.polygon.contains(point):
                return estimated_cell

        except IndexError:
            if not oob_ok:
                msg = f'Point ({x_m}, {y_m}) is outside the grid.'
                self.logger.log_message(f"Following error occurred in 'FireSim.get_cell_from_xy()': {msg}")
                raise ValueError(msg)

            return None

        # Check neighboring cells
        for neighbor in estimated_cell.neighbors:
            neighbor_cell = self._cell_dict[neighbor[0]]

            if neighbor_cell.polygon.contains(point):
                return neighbor_cell

        # If no cell contains the point and oob_ok is False, raise an error
        if not oob_ok:
            msg = f'Point ({x_m}, {y_m}) is outside the grid.'

            self.logger.log_message(f"Following error occurred in 'FireSim.get_cell_from_xy()': {msg}")
            raise ValueError(msg)

        return None

    def get_cell_from_indices(self, row: int, col: int) -> Cell:
        """Returns the cell in the sim at the indices [row, col] in 
           :py:attr:`~fire_simulator.fire.FireSim.cell_grid`.
        
        Columns increase left to right in the sim visualization window, rows increase bottom to
        top.

        :param row: row index of the desired cell
        :type row: int
        :param col: col index of the desired cell
        :type col: int
        :raises TypeError: if row or col is not of type int
        :raises ValueError: if row or col is out of the array bounds
        :return: :class:`~fire_simulator.cell.Cell` instance at the indices [row, col] in the 
                 :py:attr:`~fire_simulator.fire.FireSim.cell_grid`.
        :rtype: :class:`~fire_simulator.cell.Cell`
        """
        if not isinstance(row, int) or not isinstance(col, int):
            msg = (f"Row and column must be integer index values. "
                f"Input was {type(row)}, {type(col)}")

            self.logger.log_message(f"Following erorr occurred in 'FireSim.get_cell_from_indices(): "
                                    f"{msg} Program terminated.")
            raise TypeError(msg)

        if col < 0 or row < 0 or row >= self._grid_height or col >= self._grid_width:
            msg = (f"Out of bounds error. {row}, {col} "
                f"are out of bounds for grid of size "
                f"{self._grid_height}, {self._grid_width}")

            self.logger.log_message(f"Following erorr occurred in 'FireSim.get_cell_from_indices(): "
                                    f"{msg} Program terminated.")
            raise ValueError(msg)

        return self._cell_grid[row, col]

    # Functions for setting state of cells
    def set_state_at_xy(self, x_m: float, y_m: float, state: CellStates):
        """Set the state of the cell at the point (x_m, y_m) in the Cartesian plane.

        :param x_m: x position of the desired point in meters
        :type x_m: float
        :param y_m: y position of the desired point in meters
        :type y_m: float
        :param state: desired state to set the cell to (:py:attr:`CellStates.FIRE`,
                      :py:attr:`CellStates.FUEL`, or :py:attr:`CellStates.BURNT`) if set to
                      :py:attr:`CellStates.FIRE`, fire_type will default to
                      :py:attr:`FireTypes.WILD`
        :type state: :class:`~utilities.fire_util.CellStates`
        """
        cell = self.get_cell_from_xy(x_m, y_m)
        self.set_state_at_cell(cell, state)

    def set_state_at_indices(self, row: int, col: int, state: CellStates):
        """Set the state of the cell at the indices [row, col] in 
        :py:attr:`~fire_simulator.fire.FireSim.cell_grid`.

        Columns increase left to right in the sim window, rows increase bottom to top.

        :param row: row index of the desired cell
        :type row: int
        :param col: col index of the desired cell
        :type col: int
        :param state: desired state to set the cell to (:py:attr:`CellStates.FIRE`,
                      :py:attr:`CellStates.FUEL`, or :py:attr:`CellStates.BURNT`) if set to
                      :py:attr:`CellStates.FIRE`, fire_type will default to
                      :py:attr:`FireTypes.WILD`
        :type state: :class:`~utilities.fire_util.CellStates`
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_state_at_cell(cell, state)

    def set_state_at_cell(self, cell: Cell, state: CellStates, fire_type=FireTypes.WILD):
        """Set the state of the specified cell

        :param cell: :class:`~fire_simulator.cell.Cell` object whose state is to be changed
        :type cell: :class:`~fire_simulator.cell.Cell`
        :param state: desired state to set the cell to (:py:attr:`CellStates.FIRE`,
                      :py:attr:`CellStates.FUEL`, or :py:attr:`CellStates.BURNT`) if set to
                      :py:attr:`CellStates.FIRE`, fire_type will default to
                      :py:attr:`FireTypes.WILD`
        :type state: :class:`~utilities.fire_util.CellStates`
        :param fire_type: type of fire, only relevant if state = :py:attr:`CellStates.FIRE`,
                          options: (:py:attr:`FireTypes.WILD`, :py:attr:`FireTypes.PRESCRIBED`),
                          defaults to :py:attr:`FireTypes.WILD`
        :type fire_type: :class:`~utilities.fire_util.FireTypes`, optional
        :raises TypeError: if 'cell' is not of type :class:`~fire_simulator.cell.Cell`
        :raises ValueError: if 'cell' is not a valid :class:`~fire_simulator.cell.Cell` in the 
                            current fire Sim
        :raises TypeError: if 'state' is not a valid :class:`~utilities.fire_util.CellStates` value
        :raises TypeError: if 'fire_type' is not a valid :class:`~utilities.fire_util.FireTypes`
                           value
        """
        if not isinstance(cell, Cell):
            msg = f"'cell' must be of type 'Cell' not {type(cell)}"

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_state_at_cell(): "
                                    f"{msg} Program terminated.")
            raise TypeError(msg)

        if cell.id not in self._cell_dict:
            msg = f"{cell} is not a valid cell in the current fire Sim"

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_state_at_cell(): "
                                    f"{msg} Program terminated.")
            raise ValueError(msg)

        if not isinstance(state, int) or 0 > state > 2:
            msg = (
                f"{state} is not a valid cell state. Must be of type CellStates. "
                f"Valid states: fireUtil.CellStates.BURNT, fireUtil.CellStates.FUEL, "
                f"fireUtil.CellStates.FIRE or 0, 1, 2"
            )

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_state_at_cell(): "
                                    f"{msg} Program terminated.")
            raise TypeError(msg)

        if not isinstance(fire_type, int) or 0 > fire_type > 1:
            msg = (f"{fire_type} is not a valid fire type. Must be of type int. "
                f"Valid states: fireUtil.FireTypes.WILD, fireUtil.FireTypes.PRESCRIBED or 0, 1")

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_state_at_cell(): "
                                    f"{msg} Program terminated.")
            raise TypeError(msg)

        # Remove cell from data structures related to previous state
        prev_state = cell.state

        if prev_state == CellStates.FIRE:
            self._curr_fires_anti_cache.append(cell)

        elif prev_state == CellStates.BURNT:
            self._burnt_cells.remove(cell)

        # Set new state
        if state == CellStates.FIRE:
            cell._set_fire_type(fire_type)
            self._curr_fires_cache.append(cell)

        elif state == CellStates.FUEL:
            pass

        elif state == CellStates.BURNT:
            self._burnt_cells.add(cell)

        cell._set_state(state)


    # Functions for setting wild fires
    def set_wild_fire_at_xy(self, x_m: float, y_m: float):
        """Set a wild fire in the cell at position (x_m, y_m) in the Cartesian plane.

        :param x_m: x position of the desired wildfire ignition point in meters
        :type x_m: float
        :param y_m: y position of the desired wildfire ignition point in meters
        :type y_m: float
        """
        cell = self.get_cell_from_xy(x_m, y_m)
        self.set_wild_fire_at_cell(cell)

    def set_wild_fire_at_indices(self, row: int, col: int):
        """Set a wild fire in the cell at indices [row, col] in 
        :py:attr:`~fire_simulator.fire.FireSim.cell_grid`

        :param row: row index of the desired wildfire ignition cell
        :type row: int
        :param col: col index of the desired wildfire ignition cell
        :type col: int
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_wild_fire_at_cell(cell)

    def set_wild_fire_at_cell(self, cell: Cell):
        """Set a wild fire at a specific cell

        :param cell: :class:`~fire_simulator.cell.Cell` object to set a wildfire in
        :type cell: :class:`~fire_simulator.cell.Cell`
        """
        self.set_state_at_cell(cell, CellStates.FIRE, fire_type=FireTypes.WILD)

    # Functions for setting prescribed fires
    def set_prescribed_fire_at_xy(self, x_m: float, y_m: float):
        """Set a prescribed fire in the cell at position x_m, y_m in the Cartesian plane.

        :param x_m: x position of the desired prescribed ignition point in meters
        :type x_m: float
        :param y_m: y position of the desired prescribed ignition point in meters
        :type y_m: float
        """
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok = True)

        if cell is not None:
            self.set_prescribed_fire_at_cell(cell)

    def set_prescribed_fire_at_indices(self, row: int, col: int):
        """Set a prescribed fire in the cell at indices [row, col] in the sim's backing array

        :param row: row index of the desired prescribed ignition cell
        :type row: int
        :param col: col index of the desired prescribed ignition cell
        :type col: int
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_prescribed_fire_at_cell(cell)

    def set_prescribed_fire_at_cell(self, cell: Cell):
        """Set a prescribed fire at a specific cell.

        :param cell: :class:`~fire_simulator.cell.Cell` object to set a prescribed fire in
        :type cell: :class:`~fire_simulator.cell.Cell`
        """
        self.set_state_at_cell(cell, CellStates.FIRE, fire_type=FireTypes.PRESCRIBED)

    # Functions for setting fuel content
    def set_fuel_content_at_xy(self, x_m: float, y_m: float, fuel_content: float):
        """Set the fraction of fuel remaining at a point (x_m, y_m) in the Cartesian plane between
        0 and 1.

        :param x_m: x position in meters of the point where fuel content should be changed 
        :type x_m: float
        :param y_m: y position in meters of the point where fuel content should be changed
        :type y_m: float
        :param fuel_content: desired fuel content at point (x_m, y_m) between 0 and 1. 
        :type fuel_content: float
        """
        cell = self.get_cell_from_xy(x_m, y_m)
        self.set_fuel_content_at_cell(cell, fuel_content)

    def set_fuel_content_at_indices(self, row: int, col: int, fuel_content: float):
        """Set the fraction of fuel remanining in the cell at indices [row, col] in 
        :py:attr:`~fire_simulator.fire.FireSim.cell_grid` between 0 and 1.

        :param row: row index of the cell where fuel content should be changed
        :type row: int
        :param col: col index of the cell where fuel content should be changed
        :type col: int
        :param fuel_content: desired fuel content at indices [row, col} between 0 and 1.
        :type fuel_content: float
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_fuel_content_at_cell(cell, fuel_content)

    def set_fuel_content_at_cell(self, cell: Cell, fuel_content: float):
        """Set the fraction of fuel remaining in a cell between 0 and 1

        :param cell: :class:`~fire_simulator.cell.Cell` object to set fuel content in
        :type cell: :class:`~fire_simulator.cell.Cell`
        :param fuel_content: desired fuel content at cell between 0 and 1.
        :type fuel_content: float
        :raises TypeError: if 'cell' is not of type :class:`~fire_simulator.cell.Cell`
        :raises ValueError: if 'cell' is not a valid :class:`~fire_simulator.cell.Cell` in the
                            current sim
        :raises ValueError: if 'fuel_content' is not between 0 and 1
        """
        if not isinstance(cell, Cell):
            msg = f"'cell' must be of type Cell not {type(cell)}"

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_content_at_cell(): "
                                    f"{msg} Program terminated.")
            raise TypeError(msg)

        if cell.id not in self._cell_dict:
            msg = f"{cell} is not a valid cell in the current fire Sim"

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_content_at_cell(): "
                                    f"{msg} Program terminated.")
            raise ValueError(msg)

        if fuel_content < 0 or fuel_content > 1:
            msg = (f"'fuel_content' must be a float between 0 and 1. "
                f"{fuel_content} was provided as input")

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_content_at_cell(): "
                                    f"{msg} Program terminated.")
            raise ValueError(msg)

        cell._set_fuel_content(fuel_content)

        # Add cell to update dictionary
        self._updated_cells[cell.id] = cell

    # Functions for setting fuel moisture
    def set_fuel_moisture_at_xy(self, x_m: float, y_m: float, fuel_moisture: float):
        """Set the fuel moisture at the point (x_m, y_m) in the Cartesian plane.

        :param x_m: x position in meters of the point where fuel moisture is set
        :type x_m: float
        :param y_m: y position in meters of the point where fuel moisture is set
        :type y_m: float
        :param fuel_moisture: desired fuel moisture at point (x_m, y_m), between 0 and 1.
        :type fuel_moisture: float
        """
        cell = self.get_cell_from_xy(x_m, y_m)
        self.set_fuel_moisture_at_cell(cell, fuel_moisture)

    def set_fuel_moisture_at_indices(self, row: int, col: int, fuel_moisture: float):
        """Set the fuel moisture at the cell at indices [row, col] in the sim's backing array.

        :param row: row index of the cell where fuel moisture is set
        :type row: int
        :param col: col index of the cell where fuel moisture is set
        :type col: int
        :param fuel_moisture: desired fuel moisture at indices [row, col], between 0 and 1.
        :type fuel_moisture: float
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_fuel_moisture_at_cell(cell, fuel_moisture)

    def set_fuel_moisture_at_cell(self, cell: Cell, fuel_moisture: float):
        """Set the fuel mositure at a cell

        :param cell: cell where fuel moisture is set
        :type cell: :class:`~fire_simulator.cell.Cell`
        :param fuel_moisture: desired fuel mositure at cell, between 0 and 1.
        :type fuel_moisture: float
        :raises TypeError: if 'cell' is not of type :class:`~fire_simulator.cell.Cell`
        :raises ValueError: if 'cell' is not a valid :class:`~fire_simulator.cell.Cell` in the
                            current sim
        :raises ValueError: if 'fuel_moisture' is not between 0 and 1
        """
        if not isinstance(cell, Cell):
            msg = f"'cell' must be of type Cell not {type(cell)}"

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_moisture_at_cell(): "
                                    f"{msg} Program terminated.")
            raise TypeError(msg)

        if cell.id not in self._cell_dict:
            msg = f"{cell} is not a valid cell in the current fire Sim"

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_moisture_at_cell(): "
                                    f"{msg} Program terminated.")
            raise ValueError(msg)

        if fuel_moisture < 0 or fuel_moisture > 1:
            msg = (f"'fuel_moisture' must be a float between 0 and 1. "
                f"{fuel_moisture} was provided as input")

            self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_moisture_at_cell(): "
                                    f"{msg} Program terminated.")
            raise ValueError(msg)

        cell._set_dead_m(fuel_moisture)

        # Add cell to update dictionary
        self._updated_cells[cell.id] = cell
        self._soaked.append(cell.to_log_format())

    def add_agent(self, agent: AgentBase):
        """Add agent to sim's list of registered agent.
        
        Enables sim to log agent data along with sim data so that it is included in visualizations.

        :param agent: agent to be added to the sim's list
        :type agent: :class:`~base_classes.agent_base.AgentBase`
        :raises TypeError: if agent is not an instance of :class:`~base_classes.agent_base.AgentBase`
        """
        if isinstance(agent, AgentBase):
            self._agent_list.append(agent)
            self._agents_added = True
            self.logger.log_message(f"Agent with id {agent.id} added to agent list.")
        else:
            msg = "'agent' must be an instance of 'AgentBase' or a subclass"

            self.logger.log_message(f"Following erorr occurred in 'FireSim.add_agent(): "
                                    f"{msg} Program terminated.")
            raise TypeError(msg)

    def get_curr_wind_vec(self) -> list:
        """Get the current wind conditions as its velocity components.

        :return: The current wind conditions as a 2d vector [x velocity (m/s), y velocity (m/s)]
        :rtype: list
        """

        return self.wind_vec.vec

    def get_curr_wind_speed_dir(self) -> Tuple[float, float]:
        """Get the current wind conditions broken up into a speed (m/s) and a direction (deg).

        :return: The current wind conditions as a tuple in the form of (speed(m/s), direction(deg))
        :rtype: Tuple[float, float]
        """
        return self.wind_vec.wind_speed, self.wind_vec.wind_dir_deg

    @property
    def cell_grid(self) -> np.ndarray:
        """2D array of all the cells in the sim at the current instant.
        """
        return self._cell_grid

    @property
    def grid_width(self) -> int:
        """Width of the sim's backing array or the number of columns in the array
        """
        return self._grid_width

    @property
    def grid_height(self) -> int:
        """Height of the sim's backing array or the number of rows in the array
        """
        return self._grid_height

    @property
    def cell_dict(self) -> dict:
        """Dictionary mapping cell IDs to their respective :class:`~fire_simulator.cell.Cell` instances.
        """
        return self._cell_dict

    @property
    def iters(self) -> int:
        """Number of iterations run so far by the sim
        """
        return self._iters

    @property
    def curr_time_s(self) -> int:
        """Current sim time in seconds
        """
        return self._curr_time_s

    @property
    def curr_time_m(self) -> float:
        """Current sim time in minutes
        """
        return self.curr_time_s/60

    @property
    def curr_time_h(self) -> float:
        """Current sim time in hours
        """
        return self.curr_time_m/60

    @property
    def time_step(self) -> int:
        """Time-step of the sim. Number of seconds per iteration
        """
        return self._time_step

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the sim's backing array in (rows, cols)
        """
        return self._shape

    @property
    def size(self) -> Tuple[float, float]:
        """Size of the sim region (width_m, height_m)
        """
        return self._size

    @property
    def x_lim(self) -> float:
        """Max x coordinate in the sim's map in meters
        """
        return self.grid_width * np.sqrt(3) * self.cell_size

    @property
    def y_lim(self) -> float:
        """Max y coordinate in the sim's map in meters
        """
        return self.grid_height * 1.5 * self.cell_size
    @property
    def cell_size(self) -> float:
        """Size of each cell in the simulation.
        
        Measured as the distance in meters between two parallel sides of the regular hexagon cells.
        """
        return self._cell_size

    @property
    def sim_duration(self) -> float:
        """Duration of time (in seconds) the simulation should run for, the sim will
        run for this duration unless the fire is extinguished before the duration has passed.
        """
        return self._sim_duration

    @property
    def updated_cells(self) -> dict:
        """Dictionary containing cells updated since last time real-time visualization was updated. Dict keys
        are the ids of the :class:`~fire_simulator.cell.Cell` objects.
        """
        return self._updated_cells

    @property
    def curr_fires(self) -> list:
        """List of :class:`~fire_simulator.cell.Cell` objects that are currently in the `FIRE` state.
        """
        return self._curr_fires

    @property
    def burnt_cells(self) -> list:
        """List of :class:`~fire_simulator.cell.Cell` objects that are currently in the `BURNT` state.
        """
        return self._burnt_cells

    @property
    def roads(self) -> list:
        """List of points that define the roads for the simulation.
        
        Format for each element in list: ((x,y), fuel_content). 
        
        - (x,y) is the spatial position in the sim measured in meters
            
        - fuel_content is the amount of fuel modeled at that point (between 0 and 1)
        """
        return self._roads

    @property
    def fire_break_cells(self) -> list:
        """List of :class:`~fire_simulator.cell.Cell` objects that fall along fire breaks
        """
        return self._fire_break_cells

    @property
    def fire_breaks(self) -> list:
        """List of dictionaries representing fire-breaks.
        
        Each dictionary has:
        
        - a "geometry"  key with a :py:attr:`shapely.LineString`
        - a "fuel_value" key with a float value which represents the amount of fuel modeled along the :py:attr:`LineString`.
        """
        return self._fire_breaks

    @property
    def finished(self) -> bool:
        """`True` if the simulation is finished running. `False` otherwise
        """
        return self._finished


    @property
    def topography_map(self) -> np.ndarray:
        """2D array that represents the elevation in meters at each point in space
        """
        return self._topography_map

    @property
    def fuel_map(self) -> np.ndarray:
        """2D array that represents the spatial distribution of fuel types in the sim.

        Each element is one of the `13 Anderson FBFMs <https://www.fs.usda.gov/rm/pubs_int/int_gtr122.pdf>`_.
        """
        return self._fuel_map

    @property
    def wind_vec(self) -> Wind:
        """:class:`~fire_simulator.wind.Wind` object which defines the wind conditions for the current sim.
        """
        return self._wind_vec

    @property
    def topography_res(self) -> float:
        """Resolution of the topography map in meters
        """
        return self._topography_res

    @property
    def fuel_res(self) -> float:
        """Resolution of the fuel map in meters
        """
        return self._fuel_res

    @property
    def initial_ignition(self) -> list:
        """List of shapely polygons that were initially ignited at the start of the sim
        """
        return self._initial_ignition
