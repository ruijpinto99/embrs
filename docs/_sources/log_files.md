# Log Files
Once your simulation(s) has finished running, the log folder you specified will now be populated with the log files for that simulation.

## Folder Structure

Here's an overview of the log folder file structure:

```
Log Folder (Directory input when running sim)
├── log_dd-mmm-yyyy-hh-mm-ss 
│   ├── run_0 
│   │   ├── agents.msgpack
│   │   ├── log.msgpack 
│   │   └── status_log.json 
│   ├── run_1
│   │   ├── agents.msgpack
│   │   ├── log.msgpack 
│   │   └── status_log.json
│   ├──init_fire_state.pkl
├── ...

```

```{warning}
Do not change the folder structure of the logs! This will affect the ability to run visualizations and properly access the data
```
## Log Folder
- The log folder you select can contain as many individual logs as you would like.
- Each log within a log folder will be automatically dated as shown in the structure above.

## Initial Fire State Log
- In the top level of a log will be a file called `init_fire_state.pkl`.
- This file captures the initial state of the sim which will be the same for each run of a simulation
- It is used by the visualization tool to load the initial state of each sim
- You would also need to load this file if trying to reconstruct a state at a certain time step of the sim (see example x)

## Run Folder
- Each simulation run gets a dedicated folder within a log that is numbered by the order of execution
- All the data specific to that simulation run is stored in this folder

## Simulation Log File
- Within the run folder you will find the simulation log file titled `log.msgpack`. This file stores the changes to the state at each time-step
- This is the file that is loaded by the visualization tool to replay the states of the simulation over time.
- If you would like to reconstruct a state at a certain time step you would need to load this file [see the example code.](examples:sim_logs)

## Status Logs
Also within the run folder you will find the status log file title `status_log.json`. This file is a human-readable log file with the following information:

### Sim Start
At the top of the status log is the date and time that the sim was started, which will look like:

    ```
    {
        "sim start": "2023-08-07 11:54:25",
        ...
    }
    ```

### Metadata
Next, you will see the sim's metadata. This includes the input that were selected when running the sim along with the map file it was run on.

```
{
...
    "metadata": {
        "inputs": {
            "cell size": 10,
            "time step (sec)": 5,
            "duration (sec)": 3600,
            "roads imported": true
        },
        "sim size": {
            "rows": 712,
            "cols": 708,
            "total cells": 504096,
            "width (m)": 3500,
            "height (m)": 3500
        },
        "wind_forecast": {
            "file location": "/path/to/map/folder/wind.npy",
        },
        "imported_code": {
            "imported module location": "/path/to/imported/module.py
            "imported class name": "ImportedClassName"
        },
        "map": {
            "map file location": "/path/to/map/folder/tutorial_map.json",
            "map contents": {
                "geo_bounds": {
                    "south": 48.107,
                    "north": 48.2066,
                    "west": -120.5224,
                    "east": -120.3553
                },
                "fuel": {
                    "file": "/path/to/map/folder/fuel.npy",
                    "width_m": 12270,
                    "height_m": 10680,
                    "rows": 356,
                    "cols": 409,
                    "resolution": 30,
                    "uniform": false
                },
                "topography": {
                    "file": "/path/to/map/folder/topography.npy",
                    "width_m": 12270,
                    "height_m": 10680,
                    "rows": 10740,
                    "cols": 12300,
                    "resolution": 1.0,
                    "uniform": false
                },
                "roads": {
                    "file": "/path/to/map/folder/roads.pkl"
                },
                "initial_ignition": [
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                ...
                            ]
                        ]
                    }
                ],
                "fire_breaks": [
                    {
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [
                                    ...
                                ]
                            ]
                        },
                        "fuel_value": 10.0
                    }
            }
        }
    },
    ...
}
```

### Messages
Next, you will see some time-stamped messages providing different status updates throughout the simulation. You can add custom messages that will be published here (see [example code](examples:logging)).

```
{
    ...
    "messages": [
        "[2023-08-07 11:54:25]:Initialization successful, simulation started. ",
        "[2023-08-07 11:55:42]:Simulation complete. Sim time: 1 h 0 min, took 1 min 16 s seconds to compute. ",
        "[2023-08-07 11:56:08]:Log data saved successfully! "
    ],
}
    
```

### Results
Finally, there is a results portion of the file that provides data on whether the sim was interrupted, how many cells burnt, how much area burned, whether the fire was extinguished, how many cells are burning at the end of the simulation, and how much area is still burning.

```
{
    ...
    "results": {
        "user interrupted": false,
        "cells burnt": 115,
        "burnt area (m^2)": 29877.876430563134,
        "fire extinguished": false,
        "burning cells remaining": 1889,
        "burning area remaining (m^2)": 490776.5963246414
}
    }
}
```

## Agent Log File  
- If your imported user code registered any agents with the sim ([see Custom Control Classes: Agents](user_code:agents)) you will also see the `agents.msgpack` file inside the run folder.
- This captures the location of all agents during the simulation.
- This file will be loaded by the visualization tool to display agent locations during the simulation.