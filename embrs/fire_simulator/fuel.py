"""Representation of fuel types for the simulator

.. autoclass:: Fuel
    :members:
"""

from embrs.utilities.fire_util import FuelConstants as fc
from embrs.utilities.fire_util import CellStates

class Fuel:
    """Class representation of a fuel type, contains various relevant fuel properites to model
    fire spread

    :param fuel_type: numerical value of fuel type, corresponds to 13 Anderson FBFMs
    :type fuel_type: int
    :param init_state: initial state of the cell for given fuel type, defaults to
                       :py:attr:`CellStates.FUEL`
    :type init_state: :class:`~utilities.fire_util.CellStates`, optional
    """
    def __init__(self, fuel_type: int, init_state = CellStates.FUEL):
        """Constructor method to initialize a :class:`~fire_simulator.fuel.Fuel` instance.
        Calculates relevant properties from

        :class:`~utilities.fire_util.FuelConstants`.
        """
        self._fuel_type = fuel_type
        self._name = fc.fuel_names[fuel_type]

        # Grab all relevant fuel properties
        self._spread_probs = fc.nom_spread_prob_table[fuel_type]
        self._nominal_vel = fc.nom_spread_vel_table[fuel_type]*fc.ch_h_to_m_min
        self._consumption_factor = fc.fuel_consumption_factor_table[fuel_type]
        self._dead_m_ext = fc.dead_fuel_moisture_ext_table[fuel_type]

        self._init_state = init_state
        self._init_fuel = fc.init_fuel_table[fuel_type]

    def __str__(self) -> str:
        """To string method of :class:`fire_simulator.fuel.Fuel` class.

        :return: String representation of the fuel type.
        :rtype: str
        """
        return f"type: {self.name}"

    @property
    def fuel_type(self) -> int:
        """Numerical representation of fuel type, corresponds to 13 Anderson FBFMs.

        Possible values: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 91, 92, 93, 98, 99

        """
        return self._fuel_type

    @property
    def name(self) -> str:
        """String representing the name of the fuel type
        """
        return self._name

    @property
    def spread_probs(self) -> dict:
        """Dictionary containing the nominal spread probability from the current fuel type
        to all others. 
        
        The keys of the dictionary are the integer values representing all the fuel types. 
        The values are the spread probability between the current instance's fuel type and
        all the others.
        """
        return self._spread_probs

    @property
    def nominal_vel(self) -> float:
        """Nominal spread velocity of a wildfire in the fuel type (m/min)
        """
        return self._nominal_vel

    @property
    def consumption_factor(self) -> float:
        """Weighting factor for mass-loss curve based on 
        `(Coen 2005) <http://dx.doi.org/10.1071/WF03043>`_
        """
        return self._consumption_factor

    @property
    def dead_m_ext(self) -> float:
        """The dead fuel moisture of extinction for the fuel type
        """
        return self._dead_m_ext

    @property
    def init_state(self) -> CellStates:
        """The initial state of the fuel type
        """
        return self._init_state

    @property
    def init_fuel(self) -> float:
        """The initial amount of fuel a fuel type contains (between 0 and 1)
        """
        return self._init_fuel
