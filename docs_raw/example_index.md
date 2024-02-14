# Example Index
The EMBRS repository contains a set examples to demonstrate most of the features of the model. The examples and their contents are described below. Before going through the examples, it is recommended that you go through the *Tutorials* and read through *Theory and Background*.

The example code can be found in the folder titled 'examples'.

## Basic Functionality

(examples:sim_logs)=
### i_interpretting_sim_logs.py
This example walks through how to load data from a log file and reconstruct the state of the fire at any desired time step.

(examples:interface)=
### ii_embrs_interface.py
This example shows usage of some of the useful functions in the EMBRS interface.

Including:
- Retrieving cells
- Setting fuel content, moisture, and state
- Setting prescribed fires
- Retrieving current wind conditions

To run this example code, start a fire sim and select this file as the "User Module"

(examples:logging)=
### iii_custom_logging.py
This examples shows how to use the custom logging feature to log messages based on events and actions during a simulation.

To run this example code, start a fire sim and select this file as the "User Module"

## Prediction Tool

(examples:prediction_model)=
### iv_fire_prediction_model.py
This example shows how to construct and run a fire prediction and prints out the resulting prediction.

To run this example code, start a fire sim and select this file as the "User Module"

## Example Firefighting Code

(examples:sample_agent)=
### v_sample_agent.py
Example class which shows how to create a custom agent class from `AgentBase`. This code does not run on its own but it is used by
the next two examples to carryout firefighting operations.

(examples:controlled_burning)=
### vi_controlled_burning.py
Sample class which carries out prescribed burns based on the current wind conditions
and the locations of the fire breaks and roads.

**Note**: This example is not intended to be a robust firefighting algorithm, it is to be used
as a reference on how agents can be used in conjuction with the fire interface to formulate
a response to a fire. It does not, for example, handle the case where there is no wind.

This example works well when using the provided example map titled 'burnout_map' and the sample
wind forecast titled 'burnout_wind_forecast.json'

To run this example code, start a fire sim and select this file as the "User Module"

(examples:suppression)=
### vii_suppression.py
Sample class which carries out suppression operations based on the location of 
the frontier of the fire.

**Note**: This example is not intended to be a robust firefighting algorithm, it is to be used 
as a reference on how agents can be used in conjuction with the fire interface to formulate
a response to a fire.

To run this example code, start a fire sim and select this file as the "User Module"