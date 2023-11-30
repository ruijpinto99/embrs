# Background

EMBRS is a graphical command-line program developed in Python. Programs are started from a command line call but user input is gathered in GUI windows. 

EMBRS enables users to create realistic fire scenarios by importing real fuel and elevation data from the [LANDFIRE data base](https://www.landfire.gov/). It is built to be used with independent code developed by the user that can interact with the fire in real-time, carrying out operations such as prescribed burns and fuel soaking. This functionality provides the opportunity to get closed-loop feedback of firefighting actions while a fire is evolving. The simulations can be visualized in real-time while they are running, or visualized from the automatically generated log files. Users can also perform any desired post-processing on the simulations and get the full state of the fire at any time step.

# General Workflow

There are three primary tools used to facilitate the software's workflow:

## Map Generator:

The map generator tool is what generates all the necessary files to create a map that is usable by a simulation. During this process the user is able to import the fuel map and topography from a region in the United States ([see Generating Raw Data](./raw_data.md)) or can define a custom fuel map with flat topography. The initial ignition region and any fuel-breaks are also specified through this process ([See Creating a Map](./map_creation.md)).

## Sim:

The sim tool is used when the user is ready to run a simulation. The tool prompts the user to select a map file, a wind forecast, a location to save the log files, the user module/class to run and a series of sim parameters. Once finished selecting all parameters the simulation(s) will run ([see Running a Simulation](./running_sim.md)).

## Visualization Tool:

The visualization tool is used for visualizing a simulation. When run, the user is prompted to select the log file of interest along with some display parameters. Once all parameters have been entered the visualization of the simulation run will run in a loop ([see Sim Visualization](./visualization.md)).

# Sim Inputs / EMBRS Maps
EMBRS simulations require a map file to run. Maps contain all pertinent information that will affect the spread and characteristics of the fire. The following inputs are what define an EMBRS map:

### Fuel Map

The fuel map describes the types of vegetation within the map. The fuels in the fuel map are derived from [Anderson's 13 Fire Behavior Fuel models](https://www.fs.usda.gov/rm/pubs_int/int_gtr122.pdf). Each of these fuels have different burning characteristics that are detailed in [Fuels](fire_modelling:fuels). The fuel map can be a uniform map of a single fuel type or real fuel data can be imported from the LANDFIRE database.

### Elevation Map

The elevation map captures the topography of the map region. The slope of the landscape plays a massive role in the spread of wildfires as fire spreads more 
quickly uphills and more slowly downhill. Elevation data should be specified in meters above sea level at each location. The elevation map can be flat or real elevation data can be imported from the LANDFIRE database.

### Initial Ignition(s)

The initial ignition(s) is one or more polygon region of the map that will be on fire when the simulation starts. This is essentially the initial state of the fire when the scenario begins.

### Roads (optional)

Roads are an optional component of an EMBRS map. Roads are often leveraged when fighting fires as points where it may be easier to control a front. The roads in EMBRS are modeled to reflect the reduced chance fire spread across them ([see Roads](fire_modelling:roads)). Roads can easily be imported from OpenStreetMap during the map creation process ([see Creating a Map](./map_creation.md)).

### Fuel-breaks (optional)
Fuel-breaks are an optional component of an EMBRS map. These are lines where the fuel content has been reduced by human intervention. The fuel-breaks consist of the lines in space where the fuel is reduced along with a percentage of fuel remaining along those lines. These can be used as a way to simulate man-made fuel break such as dozer lines ([see Fire-breaks](fire_modelling:fire_breaks)).


### Sample Maps
Several sample maps are provided in the EMBRS repository. These are located in the 'SampleMaps' folder. In order to use them you must open the .json file and adjust the paths to the fuel, elevation, and road data to reflect the path to where you have pulled the repository. (See [Map Files](map_creation:files))
