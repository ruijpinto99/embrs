"""Module containing abstract control class

.. autoclass:: ControlClass
    :members:

"""

from abc import ABC, abstractmethod
from embrs.fire_simulator.fire import FireSim

class ControlClass(ABC):
    """Abstract base class that custom control code must implement to be used with the simulator.
    """

    @abstractmethod
    def process_state(self, fire: FireSim) -> None:
        """This method is called after each iteration of the simulation.
        The current :class:`~fire_simulator.fire.FireSim` object is passed in.
        This is where any changes to the fire's state or firefighting actions should be made.

        :param fire: The current :class:`~fire_simulator.fire.FireSim` instance running
        :type fire: :class:`~fire_simulator.fire.FireSim`
        """
