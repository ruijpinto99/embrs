"""Representation of the discrete cells that make up the fire sim.

.. autoclass:: Cell
    :members:

"""
import numpy as np
from shapely.geometry import Polygon

from embrs.utilities.fire_util import ControlledBurnParams
from embrs.utilities.fire_util import CellStates, FireTypes
from embrs.utilities.fire_util import HexGridMath as hex
from embrs.fire_simulator.fuel import Fuel

class Cell:
    """Class representation of the discrete cells that make up the fire sim. This holds the state
    of each cell as well as the relevant fire spread parameters. Cells are regular hexagons in the
    "point-up" orientation.

    :param id: unique id of the cell within the sim
    :type id: int
    :param col: column index of the cell within the :py:attr:`~fire_simulator.fire.FireSim.cell_grid`
    :type col: int
    :param row: row index of the cell within the :py:attr:`~fire_simulator.fire.FireSim.cell_grid`
    :type row: int
    :param cell_size: size of the cell, edge length of the hexagons, units: meters
    :type cell_size: float
    :param z: elevation of the cell measured in meters, defaults to 0.0
    :type z: float, optional
    :param fuel_type: type of fuel present at the cell, can be any of the 13 Anderson FBFMs,
                      defaults to Fuel(1)
    :type fuel_type: :class:`~fire_simulator.fuel.Fuel`, optional
    """

    def __init__(self, id: int, col: int, row: int, cell_size: float, z = 0.0, fuel_type=Fuel(1)):
        """Constructor method to initialize a cell instance. Calculates x,y position in Cartesian
        plane.
        """
        self.id = id

        # Set cell indices
        self._col = col
        self._row = row

        # z is the elevation of cell in m
        self._z = z

        self._cell_size = cell_size # defined as the edge length of hexagon
        self._cell_area = self.calc_cell_area()

        # x_pos, y_pos are the global position of cell in m
        if row % 2 == 0:
            self._x_pos = col * cell_size * np.sqrt(3)
        else:
            self._x_pos = (col + 0.5) * cell_size * np.sqrt(3)

        self._y_pos = row * cell_size * 1.5

        self._fuel_type = fuel_type
        self._fuel_content = fuel_type.init_fuel
        self.fuel_at_ignition = None
        self.W = None
        self.ignition_clock = 0

        # Variables to track how long a given cell will take to burnout
        self._t_d = None

        # Variable that tracks which type of fire a burning cell is
        self._fire_type = -1

        self._state = fuel_type.init_state
        self.cont_state = 0 # continuous state, only used for FirePrediction model

        self.has_neighbor = True
        self._neighbors = []
        self._burnable_neighbors = []

        # dead fuel moisture at this cell, value based on Anderson fuel model paper
        self._dead_m = 0.08

        self.polygon = self.to_polygon()
        self.changed = False
        self.mapping = None

        if self._row % 2 == 0:
            self.mapping = hex.even_neighborhood_mapping

        else:
            self.mapping = hex.odd_neighborhood_mapping

    def _set_elev(self, z: float):
        """Set the elevation of a cell

        :param z: elevation in meters
        :type z: float
        """

        self._z = z

    def _set_fuel_content(self, fuel_content: float):
        """Set the fuel content remaining in a cell.

        :param fuel_content: amount of fuel remaining in a cell, ranging from 0 to 1
        :type fuel_content: float
        """
        self._fuel_content = fuel_content
        self.changed = True

    def _set_fuel_type(self, fuel_type: Fuel):
        """Set the fuel type of a cell.

        :param fuel_type: Fuel type, one of the 13 Anderson FBFMs
        :type fuel_type: :class:`~fire_simulator.fuel.Fuel`
        """
        self._fuel_type = fuel_type
        self._fuel_content = fuel_type.init_fuel
        self._state = fuel_type.init_state
        self.changed = True

    def _set_vprop(self, v_prop: float):
        """Calculates the time a fire will take to propagate across a cell from the propagation
        velocity

        :param v_prop: propagation velocity of the fire in m/s
        :type v_prop: float
        """

        # Calculate time it takes for fire to propagate across cell
        self._t_d = (2 * self.cell_size) / v_prop # distance across corners used as the distance

    def _set_state(self, state: CellStates):
        """Set the state of the cell to :py:attr:`CellStates.FUEL`, :py:attr:`CellStates.FIRE`,
        or :py:attr:`CellStates.BURNT`

        :param state: state of the cell
        :type state: :class:`~utilities.fire_util.CellStates`
        """
        self._state = state
        self.changed = True

        if state == CellStates.FIRE:
            if self.fuel_type.fuel_type <= 13: # Make sure cell is combustible type
                prop_speed = self.fuel_type.nominal_vel/60
                self.W = self.fuel_type.consumption_factor

                if self.fire_type == FireTypes.PRESCRIBED:
                    prop_speed *= ControlledBurnParams.nominal_vel_adj
                    self.W *= ControlledBurnParams.consumption_factor_adj

                self._set_vprop(prop_speed)

                if self.fuel_content == 1:        
                    self.ignition_clock = 0 - self._t_d # start ignition clock when fire has propagated across entire cell
                    self.fuel_at_ignition = self.fuel_content
                else:
                    adjusted_ignition_clock = -self.fuel_type.consumption_factor * np.log(self.fuel_content)
                    self.ignition_clock = 0 - adjusted_ignition_clock
                    self.fuel_at_ignition = self.fuel_content

    def _set_fire_type(self, fire_type: FireTypes):
        """Set the type of fire at a cell

        :param fire_type: Fire type to set, either :py:attr:`FireTypes.WILD` or :py:attr:`FireTypes.PRESCRIBED`
        :type fire_type: :class:`~utilities.fire_util.FireTypes`
        """
        self._fire_type = fire_type
        self.changed = True

    def _set_dead_m(self, dead_m: float):
        """Set the dead fuel moisture at a cell.
        
        Higher fuel moisture will slow the propagation of fire

        :param dead_m: dead fuel moisture as a fraction (0 to 1)
        :type dead_m: float
        """
        self._dead_m = dead_m
        self.changed = True

    def __str__(self):
        """To string method for cells.

        :return: formatted string representing the cell
        :rtype: str
        """
        return (f"(id: {self.id}, {self.x_pos}, {self.y_pos}, {self.z}, "
                f"type: {self.fuel_type.name}, "
                f"fuel_content: {self.fuel_content}, "
                f"state: {self.state}")

    def calc_cell_area(self):
        """Calculate the area of a cell in meters squared

        :return: area of a cell given in meters squared
        :rtype: float
        """
        area_m2 = (3 * np.sqrt(3) * self.cell_size ** 2) / 2
        return area_m2

    def to_polygon(self):
        """Generate a Shapely polygon representation of the hexagonal cell.

        :return: A Shapely polygon representing the hexagonal cell
        :rtype: shapely.geometry.Polygon
        """
        l = self.cell_size
        x, y = self.x_pos, self.y_pos

        # Define the vertices for the hexagon in point-up orientation
        hex_coords = [
            (x, y + l),
            (x + (np.sqrt(3) / 2) * l, y + l / 2),
            (x + (np.sqrt(3) / 2) * l, y - l / 2),
            (x, y - l),
            (x - (np.sqrt(3) / 2) * l, y - l / 2),
            (x - (np.sqrt(3) / 2) * l, y + l / 2),
            (x, y + l)  # Close the polygon
        ]

        return Polygon(hex_coords)

    def to_log_format(self):
        """Returns a dictionary of cell data at the current instant.
        Contains all data necessary to capture changes to the state of a cell over time. 
        Used to create logs of the sim for visualization or post-processing use.

        :return: **Dictionary with the following fields:**

                - "id": (int) representing the id of the cell object
                - "state": (:class:`~utilities.fire_util.CellStates`) representing the current state of the cell
                - "fuel_content": (float) representing the amount of fuel remaining in the cell
        
                **If the cell's state is on fire the dictionary will also include:**
                    
                - "fire_type": (:class:`~utilities.fire_util.FireTypes`) representing type of fire in the cell

        :rtype: dict
        """
        cell_data = {
            "id": self.id,
            "state": self._state,
            "fuel_content": self._fuel_content,
            "dead_m": self._dead_m
        }

        if self.state == CellStates.FIRE:
            cell_data['fire_type'] = self._fire_type

        return cell_data

    # ------ Compare operators overloads ------ #
    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    @property
    def col(self) -> int:
        """Column index of the cell within the :py:attr:`~fire_simulator.fire.FireSim.cell_grid`
        """
        return self._col

    @property
    def row(self) -> int:
        """Row index of the cell within the :py:attr:`~fire_simulator.fire.FireSim.cell_grid`
        """
        return self._row

    @property
    def cell_size(self) -> float:
        """Size of the cell in meters.
        
        Measured as the side length of the hexagon.
        """
        return self._cell_size

    @property
    def cell_area(self) -> float:
        """Area of the cell, measured in meters squared.
        """
        return self._cell_area

    @property
    def x_pos(self) -> float:
        """x position of the cell within the sim, measured in meters.
        
        x values increase left to right in the sim visualization window
        """
        return self._x_pos

    @property
    def y_pos(self) -> float:
        """y position of the cell within the sim, measured in meters.
        
        y values increase bottom to top in the sim visualization window
        """
        return self._y_pos

    @property
    def z(self) -> float:
        """Elevation of the cell measured in meters
        """
        return self._z

    @property
    def fuel_type(self) -> Fuel:
        """Type of fuel present at the cell.
        
        Can be any of the 13 Anderson FBFMs
        """
        return self._fuel_type

    @property
    def fuel_content(self) -> float:
        """Fraction of fuel remaining at the cell, between 0 and 1
        """
        return self._fuel_content

    @property
    def t_d(self) -> float:
        """Estimate of the time (in seconds) it will take the current fire to propagate across the
        cell. 

        `-1` if the cell is not on fire
        """
        return self._t_d

    @property
    def state(self) -> CellStates:
        """Current state of the cell.
        
        Can be :py:attr:`CellStates.FUEL`, :py:attr:`CellStates.BURNT`, or :py:attr:`CellStates.FIRE`.
        """
        return self._state

    @property
    def fire_type(self) -> FireTypes:
        """Current type of fire burning at the cell.
        
        Either :py:attr:`FireTypes.PRESCRIBED` or :py:attr:`FireTypes.WILD`, `-1` if not currently in state :py:attr:`CellStates.FIRE`.
        """
        return self._fire_type

    @property
    def neighbors(self) -> list:
        """List of cells that are adjacent to the cell.
        
        Each list element is in the form (id, (dx, dy))
        
        - "id" is the id of the neighboring cell
        - "(dx, dy)" is the difference between the column and row respectively of the cell and its neighbor
        """
        return self._neighbors

    @property
    def burnable_neighbors(self) -> set:
        """Set of cells adjacent to the cell which are in a burnable state.

        Each element is in the form (id, (dx, dy))
        
        - "id" is the id of the neighboring cell
        - "(dx, dy)" is the difference between the column and row respectively of the cell and its neighbor
        """
        return self._burnable_neighbors

    @property
    def dead_m(self) -> float:
        """Dead fuel moisture of the cell, as a fraction (0 to 1)
        """
        return self._dead_m
