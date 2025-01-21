"""Various sets of constants and helper functions useful throughout the codebase

.. autoclass:: UtilFuncs
    :members:

.. autoclass:: CellStates
    :members:

.. autoclass:: FireTypes
    :members:

.. autoclass:: FuelConstants
    :members:

.. autoclass:: RoadConstants
    :members:

.. autoclass:: HexGridMath
    :members:

.. autoclass:: ControlledBurnParams
    :members:

.. autoclass:: WindAdjustments
    :members:
"""

from typing import Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from functools import lru_cache

cell_type = np.dtype([
    ('id', np.int32),
    ('state', np.int32),
    ('fuelType', np.int32),
    ('fuelContent', np.float32),
    ('moisture', np.float32),
    ('position', [('x', np.float32), ('y', np.float32), ('z', np.float32)]),
    ('indices', [('i', np.int32), ('j', np.int32)]),
    ('changed', np.bool_)
])

action_type = np.dtype([
    ('type', np.int32),
    ('pos', [('x', np.float32), ('y', np.float32)]),
    ('time', np.float32),
    ('value', np.float32)
])

class SpreadDecomp:
    self_loc_to_neighbor_loc_mapping = {
        1: [(7, 'A')],
        2: [(6, 'A'), (10, 'B')],
        3: [(9, 'B')],
        4: [(8, 'B'), (12, 'C')],
        5: [(11, 'C')],
        6: [(10, 'C'), (2, 'D')],
        7: [(1, 'D')],
        8: [(12, 'D'), (4, 'E')],
        9: [(3, 'E')],
        10: [(2, 'E'), (6, 'F')],
        11: [(5, 'F')],
        12: [(4, 'F'), (8, 'A')]
    }

class CellStates:
    """Enumeration of the possible cell states.

    Attributes:
        - **BURNT** (int): Represents a cell that has been burnt and has no fuel remaining.
        - **FUEL** (int): Represents a cell that still contains fuel and is not on fire.
        - **FIRE** (int): Represents a cell that is currently on fire.
    """
    # Cell States:
    BURNT, FUEL, FIRE = 0, 1, 2

class FireTypes:
    """Enumeration of the possible fire types.

    Attributes:
        - **WILD** (int): Represents a fire that occurred naturally.
        - **PRESCRIBED** (int): Represents a fire that was started in a controlled manner, burns with lower intensity.

    """
    # Fire Types:
    WILD, PRESCRIBED = 0, 1

class FuelConstants:
    """Various values and dictionaries pertaining to modelling of fuel types.

    Attributes:
        - **burnout_thresh** (float): fuel content which dictates what is considered to be a burned out cell.
        - **ch_h_to_m_min** (float): float that converts chains/hr to m/min.
        - **fbfm_13_keys** (list): list of ints corresponding to each of Anderson's 13 FBFMs.
        - **fuel_names** (dict): dictionary where keys are ints for each FBFM and values are the names of each fuel type.
        - **fuel_type_revers_lookup** (dict): dictionary where keys are the fuel type names and values are the ints for each FBFM.
        - **init_fuel_table** (dict): dictionary of the initial fuel content of each fuel type.
        - **nom_spread_prob_table** (dict): dictionary of dictionaries for each fuel type's nominal spread probabilities.
        - **nom_spread_vel_table** (dict): dictionary of nominal spread velocities for wildfires for each fuel type.
        - **dead_fuel_moisture_ext_table** (dict): dictionary of each fuel type's dead fuel moisture of extinction.
        - **fuel_consumption_factor_table** (dict): dictionary of each fuel type's mass-loss curve weighting factor.
        - **fuel_color_mapping** (dict): dictionary mapping each fuel type to the display color for visualizations.

    """
    burnout_thresh = 0.01

    # Conversion from chains/hr to m/min
    ch_h_to_m_min = 0.33528

    # Conversion from inches to mm
    in_to_mm = 25.4

    # Valid keys for the FBFM 13 fuel model
    fbfm_13_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 91, 92, 93, 98, 99]

    # Dictionary of fuel number to name
    fuel_names = {1: "Short grass", 2: "Timber grass", 3: "Tall grass", 4: "Chaparral",
                5: "Brush", 6: "Hardwood slash", 7: "Southern rough", 8: "Closed timber litter",
                9: "Hardwood litter", 10: "Timber litter", 11: "Light logging slash",
                12: "Medium logging slash", 13: "Heavy logging slash", 91: 'Urban', 92: 'Snow/ice',
                93: 'Agriculture', 98: 'Water', 99: 'Barren'}

    fuel_type_reverse_lookup = {"Short grass": 1, "Timber grass": 2, "Tall grass": 3, "Chaparral": 4,
                "Brush": 5, "Hardwood slash": 6, "Southern rough": 7 , "Closed timber litter": 8,
                "Hardwood litter": 9, "Timber litter": 10, "Light logging slash": 11,
                "Medium logging slash": 12, "Heavy logging slash": 13, 'Urban': 91, 'Snow/ice': 92,
                'Agriculture': 93, 'Water': 98, 'Barren': 99}

    # Standard initial fuel contents for each type
    init_fuel_table = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1,
                    12: 1, 13: 1, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0}

    # Table of nominal spread probabilities between each fuel type
    nom_spread_prob_table = {
        1: {1: 0.425, 2: 0.45, 3: 0.475, 4: 0.475, 5: 0.4375, 6: 0.4625, 7: 0.425, 8: 0.425,
            9: 0.475, 10: 0.425, 11: 0.45, 12: 0.45, 13: 0.475, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        2: {1: 0.45, 2: 0.475, 3: 0.5, 4: 0.5, 5: 0.4625, 6: 0.4875, 7: 0.45, 8: 0.45, 9: 0.5,
            10: 0.45, 11: 0.475, 12: 0.475, 13: 0.5, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        3: {1: 0.475, 2: 0.5, 3: 0.525, 4: 0.525, 5: 0.4875, 6: 0.5125, 7: 0.475, 8: 0.475,
            9: 0.525, 10: 0.475, 11: 0.5, 12: 0.5, 13: 0.525, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        4: {1: 0.35, 2: 0.375, 3: 0.4, 4: 0.425, 5: 0.3875, 6: 0.4125, 7: 0.375, 8: 0.375,
            9: 0.425, 10: 0.4, 11: 0.4, 12: 0.425, 13: 0.45, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        5: {1: 0.3125, 2: 0.3375, 3: 0.3625, 4: 0.3875, 5: 0.35, 6: 0.375, 7: 0.3375, 8: 0.3375,
            9: 0.3875, 10: 0.3625, 11: 0.3625, 12: 0.3875, 13: 0.4125, 91: 0, 92: 0, 93: 0, 98: 0,
            99: 0},

        6: {1: 0.3375, 2: 0.3625, 3: 0.3875, 4: 0.4125, 5: 0.375, 6: 0.4, 7: 0.3625, 8: 0.3625,
            9: 0.4125, 10: 0.3875, 11: 0.3875, 12: 0.4125, 13: 0.4375, 91: 0, 92: 0, 93: 0, 98: 0,
            99: 0},

        7: {1: 0.3, 2: 0.325, 3: 0.35, 4: 0.375, 5: 0.3375, 6: 0.3625, 7: 0.325, 8: 0.325,
            9: 0.375, 10: 0.35, 11: 0.35, 12: 0.375, 13: 0.4, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        8: {1: 0.025, 2: 0.05, 3: 0.075, 4: 0.1, 5: 0.0625, 6: 0.0875, 7: 0.05, 8: 0.025, 9: 0.075,
            10: 0.225, 11: 0.05, 12: 0.25, 13: 0.275, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        9: {1: 0.075, 2: 0.1, 3: 0.125, 4: 0.15, 5: 0.1125, 6: 0.1375, 7: 0.1, 8: 0.075, 9: 0.125,
            10: 0.275, 11: 0.1, 12: 0.3, 13: 0.325, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        10: {1: 0.2, 2: 0.225, 3: 0.25, 4: 0.325, 5: 0.2875, 6: 0.3125, 7: 0.275, 8: 0.3, 9: 0.35,
            10: 0.3, 11: 0.325, 12: 0.325, 13: 0.35, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        11: {1: 0.05, 2: 0.075, 3: 0.1, 4: 0.125, 5: 0.0875, 6: 0.1125, 7: 0.075, 8: 0.05, 9: 0.1,
            10: 0.25, 11: 0.075, 12: 0.275, 13: 0.3, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        12: {1: 0.225, 2: 0.25, 3: 0.275, 4: 0.35, 5: 0.3125, 6: 0.3375, 7: 0.3, 8: 0.325,
            9: 0.375, 10: 0.325, 11: 0.35, 12: 0.35, 13: 0.375, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        13: {1: 0.25, 2: 0.275, 3: 0.3, 4: 0.375, 5: 0.3375, 6: 0.3625, 7: 0.325, 8: 0.35, 9: 0.4,
            10: 0.35, 11: 0.375, 12: 0.375, 13: 0.4, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        91: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        92: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        93: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        98: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},

        99: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 91: 0, 92: 0, 93: 0, 98: 0, 99: 0},
    }

    # values based on 5 mi/h wind, given in chains/hr
    nom_spread_vel_table = {1: 78, 2: 35, 3: 104, 4: 75, 5: 18, 6: 32, 7: 20, 8: 1.6,
                                    9: 7.5, 10: 7.9, 11: 6, 12: 13, 13: 13.5, 91: 0, 92: 0,
                                    93: 0, 98: 0, 99: 0}

    # based on the values in the Anderson Fuel model
    dead_fuel_moisture_ext_table = {1: 0.12, 2: 0.15, 3: 0.25, 4: 0.20, 5: 0.20, 6: 0.25, 7: 0.40,
                                    8: 0.30, 9: 0.25, 10: 0.25, 11: 0.15, 12: 0.20, 13: 0.25,
                                    91: 0.1, 92: 1, 93: 1, 98: 1, 99: 1}

    # Weighting factor for mass-loss curve for each fuel type
    fuel_consumption_factor_table = {1: 7, 2: 30, 3: 7, 4: 360, 5: 360, 6: 360, 7: 360,
                            8: 1200, 9: 1200, 10: 1200, 11: 1200, 12: 1200, 13: 1200,
                            91: 0, 92: 0, 93: 0, 98: 0, 99: 0}

    # Color mapping for each fuel type
    fuel_color_mapping = {1: 'xkcd:pale green', 2:'xkcd:lime', 3: 'xkcd:bright green',
                        4: 'xkcd:teal', 5: 'xkcd:bluish green', 6: 'xkcd:greenish teal',
                        7: 'xkcd:light blue green', 8: 'xkcd:pale olive' , 9: 'xkcd:olive',
                        10: 'xkcd:light forest green', 11: 'xkcd:bright olive',
                        12: 'xkcd:tree green', 13: 'xkcd:avocado green', 91: 'xkcd:ugly purple',
                        92: 'xkcd:pale cyan' , 93: "xkcd:perrywinkle", 98: 'xkcd:water blue',
                        99: 'xkcd:black'}


class WindAdjustments:
    """Mapping of wind speed to Lorentzian curve parameters, used to calculate alpha_w

    Attributes:
        - **wind_speed_param_mapping** (dict): dictionary mapping wind speeds (km/h) to Lorentzian parameters (A, gamma, C)
    """

    wind_speed_param_mapping = {0: (0, 1, 1), 10: (0.33, 114.93, 0.89), 20: (1.17, 84.83, 0.64), 30: (2.12, 60.95, 0.4),
                                50: (2.8, 45.13, 0.25), 60: (3.03, 39.61, 0.2), 70: (3.26, 33.92, 0.15), 80: (3.49, 27.56, 0.1),
                                90: (3.72, 20, 0.05), 100: (3.94, 7.71, 0.01)}

class ControlledBurnParams:
    """Parameters differentiating :py:attr:`~FireTypes.PRESCRIBED` fires from :py:attr:`~FireTypes.WILD`

    Attributes:
        - **nominal_prob_adj** (float): factor adjusting the nominal spread probability.
        - **nominal_vel_adj** (float): factor adjusting the nominal spread velocity.
        - **consumption_factor_adj** (float): factor adjusting the fuel consumption factor W.
        - **min_burnable_fuel_content** (float): minimum fuel fraction remaining in a neighboring cell for a possible ignition
        - **burnout_fuel_frac** (float): fraction of fuel at ignition remaining in burning cell at which it stops burning.
    """
    nominal_prob_adj = 0.75
    nominal_vel_adj = 0.5
    consumption_factor_adj = 1.5
    min_burnable_fuel_content = 0.35
    burnout_fuel_frac = 0.3

class RoadConstants:
    """Information on the modelling of different road types.

    Attributes:
        - **major_road_types** (list): list of the names of each of the major road types defined by `OpenStreetMap <https://wiki.openstreetmap.org/wiki/United_States/Road_classification>`_.
        - **road_fuel_vals** (dict): dictionary mapping road types to the amount of fuel modelled along them.
        - **road_color_mapping** (dict): dictionary mapping road types to their display color for visualizations.

    """
    # Road types
    major_road_types = ['motorway', 'trunk' , 'primary', 'secondary',
                        'tertiary', 'unclassified', 'residential']

    # Fuel values for each road type
    road_fuel_vals = {'motorway': 0, 'trunk': 0 , 'primary': 0.1, 'secondary': 0.15,
                      'tertiary': 0.2, 'unclassified': 0.3, 'residential': 0.25}

    road_color_mapping = {
        'motorway': '#4B0082',  # Indigo
        'trunk': '#800080',    # Purple
        'primary': '#9400D3',  # DarkViolet
        'secondary': '#9932CC',  # DarkOrchid
        'tertiary': '#BA55D3',  # MediumOrchid
        'unclassified': '#DA70D6',  # Orchid
        'residential': '#EE82EE',  # Violet
    }

class HexGridMath:
    """Data structures to help with handling cell neighbors in a hexagonal grid.

    Attributes:
        - **hex_angle** (float): float representing the angle between neighboring hexagon cell centers.
        - **even_neighborhood_mapping** (dict): dictionary that maps relative indices of a cell's neighbors to unit vectors between them, for even rows.
        - **odd_neighborhood mapping** (dict): dictionary that maps relative indices of a cell's neighbors to unit vectors between them, for odd rows.
        - **even_neighborhood** (list): list of relative indices of a cell's neighbors, for even rows.
        - **odd_neighborhood** (list): list of relative indices of a cell's neighbors, for odd rows.


    """
    # Define angle between neighboring hex centers
    hex_angle = 60 * (np.pi/180)

    # Define neighborhoods and unit vector mappings for even and odd rows
    even_neighborhood_mapping = {(-1,1): [-np.cos(hex_angle), np.sin(hex_angle)],
                                (0, 1): [np.cos(hex_angle), np.sin(hex_angle)],
                                (1,0): [1,0], (0, -1): [np.cos(hex_angle), -np.sin(hex_angle)],
                                (-1, -1): [-np.cos(hex_angle), -np.sin(hex_angle)],
                                (-1,0): [-1,0]}

    odd_neighborhood_mapping = {(1,0): [1,0], (1,1): [np.cos(hex_angle), np.sin(hex_angle)],
                                (0,1): [-np.cos(hex_angle), np.sin(hex_angle)], (-1,0): [-1,0],
                                (0,-1): [-np.cos(hex_angle), -np.sin(hex_angle)],
                                (1, -1): [np.cos(hex_angle), -np.sin(hex_angle)]}

    even_neighborhood = [(-1,1), (0, 1), (1,0), (0, -1), (-1, -1), (-1,0)]
    even_neighbor_letters = {'F': (-1, 1),
                            'A':(0, 1),
                            'B':(1, 0),
                            'C': (0, -1),
                            'D': (-1, -1),
                            'E': (-1, 0)}

    odd_neighborhood = [(1,0), (1,1), (0,1), (-1,0), (0,-1), (1, -1)]
    odd_neighbor_letters = {'B': (1, 0),
                            'A': (1, 1),
                            'F': (0, 1),
                            'E': (-1, 0),
                            'D': (0, -1),
                            'C': (1, -1)}

class UtilFuncs:
    """Various utility functions that are useful across numerous files.
    """
    def get_indices_from_xy(x_m: float, y_m: float, cell_size: float, grid_width: int,
                            grid_height: int) -> Tuple[int, int]:
        """Get the row and column indices in a backing array of a cell containing the point
        (x_m, y_m). 
        
        Does not require a :class:`~fire_simulator.fire.FireSim` object, uses 'cell_size' and the size of the array
        to calculate indices.

        :param x_m: x position in meters where indices should be found
        :type x_m: float
        :param y_m: y position in meters where indices should be found
        :type y_m: float
        :param cell_size: cell size in meters, measured as the distance across two parallel sides
                          of a regular hexagon
        :type cell_size: float
        :param grid_width: number of columns in the backing array of interest
        :type grid_width: int
        :param grid_height: number of rows in the backing array of interest
        :type grid_height: int
        :raises ValueError: if x or y inputs are out of bounds for the array constructed by
                            'cell_size', 'grid_width', and 'grid_height'
        :return: tuple containing [row, col] indices at the point (x_m, y_m)
        :rtype: Tuple[int, int]
        """
        row = int(y_m // (cell_size * 1.5))

        if row % 2 == 0:
            col = int(x_m // (cell_size * np.sqrt(3))) + 1
        else:
            col = int((x_m // (cell_size * np.sqrt(3))) - 0.5) + 1

        if col < 0 or row < 0 or row >= grid_height or col >= grid_width:
            msg = (f'Point ({x_m}, {y_m}) is outside the grid. '
                f'Column: {col}, Row: {row}, '
                f'simSize: ({grid_height} , {grid_width})')
            raise ValueError(msg)

        return row, col

    def get_time_str(time_s: int, show_sec = False) -> str:
        """Returns a formatted time string in m-h-s format from the time in seconds.
        
        Useful for generating readable display of the time.

        :param time_s: time value in seconds
        :type time_s: int
        :param show_sec: set to `True` if seconds should be displayed, `False` if not, defaults to `False`
        :type show_sec: bool, optional
        :return: formatted time string in h-m-s format
        :rtype: str
        """
        hours = int(time_s // 3600)
        minutes = int((time_s % 3600) // 60)

        if show_sec:
            seconds = int((time_s % 3600) % 60)

            if hours > 0:
                result = f"{hours} h {minutes} min {seconds} s"
            elif minutes > 0:
                result = f"{minutes} min {seconds} s"
            else:
                result = f"{seconds} s"
            return result

        if hours > 0:
            result = f"{hours} h {minutes} min"
        else:
            result = f"{minutes} min"
        return result

    def get_dominant_fuel_type(fuel_map: np.ndarray) -> int:
        """Finds the most commonly occurring fuel type within a fuel map.

        :param fuel_map: Fuel map for a region
        :type fuel_map: np.ndarray
        :return: Integer representation of the dominant fuel type. 
            See :py:attr:`~utilities.fire_util.FuelConstants.fuel_names`
        :rtype: int
        """
        counts = np.bincount(fuel_map.ravel())

        return np.argmax(counts)

    def get_cell_polygons(cells: list) -> list:
        """Converts a list of cell objects into the minimum number of :py:attr:`shapely.Polygon`
        required to describe all of them

        :param cells: list of :class:`~fire_simulator.cell.Cell` objects to be converted
        :type cells: list
        :return: list of :py:attr:`shapely.Polygon` representing the cells
        :rtype: list
        """

        if not cells:
            return None

        polygons = [Polygon(UtilFuncs.hexagon_vertices(cell.x_pos, cell.y_pos, cell.cell_size)) for cell in cells]

        merged_polygon = unary_union(polygons)

        if isinstance(merged_polygon, MultiPolygon):
            return list(merged_polygon.geoms)

        return [merged_polygon]

    @staticmethod
    def hexagon_vertices(x: float, y: float, s: float) -> list:
        """Calculates the locations of each of a hexagons vertices with center (x,y) and side
        length s.

        :param x: x location of hexagon center
        :type x: float
        :param y: y location of hexagon center
        :type y: float
        :param s: length of hexagon sides
        :type s: float
        :return: list of (x,y) points representing the hexagon's vertices
        :rtype: list
        """
        vertices = [
            (x, y + s),
            (x + s * np.sqrt(3) / 2, y + s / 2),
            (x + s * np.sqrt(3) / 2, y - s / 2),
            (x, y - s),
            (x - s * np.sqrt(3) / 2, y - s / 2),
            (x - s * np.sqrt(3) / 2, y + s / 2)
        ]
        return vertices
    

    def get_dist(edge_loc, idx_diff, cell_size):
        # Keys equal to difference between indices
        odd_loc_distance_dict = {
            1: cell_size / 2,
            2: (np.sqrt(3)/2) * cell_size, # Law of sines
            3: (np.sqrt(7)/2) * cell_size, # Law of cosines
            4: (3 * cell_size) / 2,
            5: (np.sqrt(13) * cell_size) / 2, # Law of cosines
            6: np.sqrt(3) * cell_size
        }

        even_loc_distance_dict = {
            2: cell_size,
            3: (np.sqrt(7)/2) * cell_size,
            4: np.sqrt(3) * cell_size,
            5: (np.sqrt(13) * cell_size) / 2,
            6: 2 * cell_size,
        }

        # Handle case where ignition starts at center
        if edge_loc == 0:
            if idx_diff % 2 == 0:
                return cell_size
            else:
                return (np.sqrt(3) * cell_size)/2

        elif edge_loc %  2 == 0:
            return even_loc_distance_dict[idx_diff]

        else:
            return odd_loc_distance_dict[idx_diff]

    @lru_cache
    def get_ign_parameters(edge_loc: int, cell_size):

        if edge_loc == 0:
            # Ignition is at the center of cell
            start_angle = 30
            end_angle = 360

            directions = np.linspace(start_angle, end_angle, 12)

            start_end_point = 1
            end_end_point = 12

        elif edge_loc % 2 == 0:
            # Ignition is at a corner point
            start_angle = (30 * edge_loc + 120) % 360
            end_angle = (start_angle + 120)

            directions = np.linspace(start_angle, end_angle, 9)

            start_end_point = (edge_loc + 2) % 12 or 12
            end_end_point = (start_end_point + 8) % 12 or 12

        else:
            # Ignition is along an edge
            start_angle = (30 * edge_loc + 90) % 360
            end_angle = (start_angle + 180)

            directions = np.linspace(start_angle, end_angle, 11)

            start_end_point = (edge_loc + 1) % 12 or 12
            end_end_point = (12 + (edge_loc - 1)) % 12 or 12

        directions = np.array([direction % 360 for direction in directions])

        if end_end_point < start_end_point:
            self_end_points = np.concatenate([
                np.arange(start_end_point, 13),
                np.arange(1, end_end_point + 1)
            ])
        
        else:
            self_end_points = np.arange(start_end_point, end_end_point + 1)

        end_points = []
        distances = []

        for end_point in self_end_points:

            idx_diff = np.abs(end_point - edge_loc)

            if idx_diff > 6:
                idx_diff = 12 - idx_diff


            dist = UtilFuncs.get_dist(edge_loc, idx_diff, cell_size)

            distances.append(dist)

            neighbor_locs = SpreadDecomp.self_loc_to_neighbor_loc_mapping[end_point]
            end_points.append(neighbor_locs)

        return np.array(directions), np.array(distances), end_points
