# EMBRS: Engineering Model for Burning and Real-time Suppression

EMBRS is a real-time fire simulation software that provides users a sandbox for testing fire
suppression algorithms and strategies in a Python framework.

## Installation
EMBRS can be installed by downloading source code or via the PyPI package manager with `pip`.

The simplest method is using `pip` with the following command:

```bash
    pip install embrs
```

Developers who would like to inspect the source code can install EMBRS by downloading the git repository from GitHub and use `pip` to install it locally. The following terminal commands can be used to do this:

```bash
    # Download source code from the 'main' branch
    git clone -b main https://github.com/AREAL-GT/embrs.git

    # Install EMBRS
    pip install -e embrs

```

## Usage
### Launching EMBRS Applications
Once EMBRS is installed, to launch the provided EMBRS applications to run a sim, run a visualization, create a map, and create a wind forecast you can use the following terminal commands.

```bash
    # Run a simulation
    run_embrs_sim
    
    
    # Run a visualization
    run_embrs_viz
    
    
    # Create an EMBRS map
    create_embrs_map
    
    
    # Create a wind forecast
    create_embrs_wind

```

Upon running these commands you will see GUI windows allowing you to specify each process. Read the rest of this site for information on how to use each.
