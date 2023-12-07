# EMBRS
Full documentation can be found at https://areal-gt.github.io/embrs/


## Installation
EMBRS can be installed by downloading source code or via the PyPI package manager with `pip`.

The simplest method is using `pip` with the following command:

```
pip install embrs
```

Developers who would like to inspect the source code can install EMBRS by downloading the git repository from GitHub and use `pip` to install it locally. The following terminal commands can be used to do this:

```
# Download source code from the 'main' branch
git clone -b main 

# install EMBRS
pip install -e embrs

```

## Usage

### Launching EMBRS Applications
Once EMBRS is installed, to launch the provided EMBRS applications to run a sim, run a visualization, create a map, and create a wind forecast you can use the following terminal commands.

```
# Run a simulation
run_embrs_sim


# Run a visualization
run_embrs_viz


# Create an EMBRS map
create_embrs_map


# Create a wind forecast
create_embrs_wind

```

Upon running these commands you will see GUI windows allowing you to specify each process. Read the full documentation for information on how to use each.

### Importing and using EMBRS clases
You will now be able to import EMBRS clases into your python files, for example if you would like to import the fire prediction module and run a prediction you could use code similar to below:
```{python}

from embrs.tools.fire_predictor import FirePredictor
from embrs.fire_simulator.fire import FireSim

#... custom class implementation


def process_state(self, fire:FireSim):
  # ... rest of process_state code


  if (some condition):
    # construct a fire predictor
      fire_predictor = FirePredictor(fire, 3, bias=1)
      prediction = fire_predictor.run_prediction()

# ... rest of custom class implementation

```

More information is provided in the documentation on code usage.
