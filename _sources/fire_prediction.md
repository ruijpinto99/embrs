# Fire Prediction Model
EMBRS ships with a fire prediction tool. This tool runs a simplified version of the core fire model with a few modifictations to predict the future propagation of a current fire over a fixed time horizon. 

The EMBRS propagation is modified in the following ways for the prediction model:
- **Homogenous fuel map**: The fuel map of the simulation is replaced with a map consisting of just a single fuel type, to model possible assumptions made when in a real fire-fighting scenario.
- **Wind uncertainty**: Error is added to the wind forecast using an auto-regressive model to model erroneous wind forecasting when making predictions.
- **Bias term introduced**: Users can choose to bias the prediction tool to either overpredict or underpredict the fire spread.
- **No post-frontal combustion modelled**: Fuel content of burning cells not modelled as the prediction only cares about ignition timing.

The fire prediction model can be found in the 'tools' folder in the file called 'fire_predictor.py'. Below is some simple example usage of this tool. For a full example of using the model see [the example code.](examples:prediction_model).

## Constructor
Before running the prediction model, a `FirePredictor` object must first be constructed. The constructor takes three arguments:

- `fire` - the `FireSim` object of the fire you would like to run a prediction on.
- `time_horizon_hr` - the time (in hours) you would like to project forward.
- `fuel_type` - the fuel type (as an int) to be used for fuel map. If -1, the fuel type defaults to the most commonly occurring fuel in `fire` fuel map.
- `bias` - bias term which controls whether the model should over or under predict. Values > 1 lead to overprediction, values < 1 lead to overprediction. Defaults to 1.
- `time_step_s` - the time-step (in seconds) you would like the prediction model to use. If `None`, defaults to twice the time step of `fire`.
- `cell_size_m` - the cell size (in meters) you would like the prediction model to use. If `None`, defaults to twice the cell size of `fire`.

**Example Constructors**

```python
from embrs.tools.fire_predictor import FirePredictor

# Construct a FirePredictor that will predict 3 hours into the future with no bias
# Using default values for fuel_type, time_step_s, and cell_size_m

pred_model = FirePredictor(fire, 3, bias=1)

# Construct a FirePredictor that will predict 2.5 hours into the future with a
# bias of 1.5, and a fuel_type of 5 (Brush)
pred_model = FirePredictor(fire, 2.5, fuel_type = 5, bias = 1.5)

# Construct a FirePredictor that will predict 3 hour into th future with a bias of
# 0.5 using a time step of 20 sec, and a cell size of 30 meters.
pred_model = FirePredictor(fire, 3, bias=0.5, time_step_s = 20, cell_size_m = 30)

```

## Running the Model
Once the `FirePredictor` object is constructed, running the prediction is simple. Just call the object's `run_prediction` method.

```python

prediction = pred_model.run_prediction()

```

```{note}
The FirePredictor instance's starting point is always based on the state of the `FireSim` object **when you inputted it as a constructor**. So if you run the prediction again after letting the `FireSim` progress, the prediction results will be based on the starting point that was originally passed in. To run another prediction on the same fire starting at a later time, you would need to construct a new `FirePredictor` object.
```

## Interpreting Results
The result of `run_prediction` is a dictionary. The dictionary's keys are each time-step that it ran. The time-steps start from the time of the `FireSim` when it was inputted to the FirePredictor model. For each of these keys, the value is a list of (x,y) coordinates where the model predicts an ignition during that \ time-step. 

For example, if the `FirePredictor` was constructed when the `FireSim` time was 1000 seconds and the time-step used was 15 seconds the output might look something like this:

```python

{
    1000: [(202.5, 506.75), (2010.0, 556.5), ...],
    1015: [(202.5, 600.25), (2010.0, 506.5), (1400.5, 800.5), ...],
    1030: [(306.45, 725.6),...],
    ...
}

```

With this output, you can understand what regions are more likely to ignite, and roughly what time they will ignite. This can be used to inform your custom control class and direct the control strategy.

Documentation for the fire prediction tool can be found [here.](./_autosummary/tools.fire_predictor.rst)