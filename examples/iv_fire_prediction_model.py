"""Demo class demonstrating how the use of the fire prediction tool during a fire simulation.
The prediction tool can be used to inform firefighting decision making.

To run this example code, start a fire sim and select this file as the "User Module"
"""

from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from embrs.tools.fire_predictor import FirePredictor

class PredictorCode(ControlClass):
    def __init__(self, fire:FireSim):
        self.prediction = False

    def process_state(self, fire: FireSim) -> None:
        if not self.prediction and fire.curr_time_h > 1:
            
            # Define time horizon to predict over
            time_horizon_hr = 2

            # Initialize the fire prediction tool
            predictor = FirePredictor(fire, time_horizon_hr, bias=1)

            # Run prediction
            future_fires = predictor.run_prediction()
            self.prediction = True

            # Print out the predicted fire locations for each time step
            for time_step in future_fires.keys():
                print(f"Time: {time_step}, fires: {future_fires[time_step]}")