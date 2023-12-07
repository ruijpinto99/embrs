# Fire Interface
The `BaseFireSim` class has a number of public functions that make up a pseudo-interface. These functions are designed to be used by custom control classes, which will have access to the current `BaseFireSim` through their `process_state` method ([See Custom Control Classes](./user_code.md)). Below are some usage examples of these public functions, for full documentation see [the dedicated documentation page](./_autosummary/base_classes.base_fire.rst)

For all the examples below we will assume that `fire` is an instance of the `BaseFireSim` class.

## Retrieving Cells
If you are interested in retrieving a specific instance of a cell object there are two ways to do so:

**From Indices:**

```python
   # row and col must be integers
   row = 10
   col = 245

   # Get the cell with indices (row, col) in the backing array
   cell = fire.get_cell_from_indices(row, col)

```

If looking at a visualization, row 0 is the row along the bottom of the visualization, and column 0 is the column along the left side of the visualization.

**From Coordinates:**

```python
   # x_m and y_m are floats in meters
   x_m = 1240.0
   y_m = 245.7

   # Get the cell that contains the point (x_m, y_m) within it
   cell = fire.get_cell_from_xy(x_m, y_m)

```

If looking at a visualization, x = 0 is along the left edge of the visualization, y = 0 is along the bottom edge of the visualization.

## Setting State
At any point, you can set the state of a cell to one of the three available [states](fire_modelling:state) (`FUEL`, `FIRE` and `BURNT`).

**This can be done by passing in the cell object explicitly:**

```python
   from utilities.fire_util import CellStates

   state = CellStates.BURNT

   # Set cell's state to BURNT
   fire.set_state_at_cell(cell, state) # cell is an instance of 'Cell' class

```

**Or by passing in the x,y coordinates:**

```python
   from utilities.fire_util import CellStates
   
   # x_m and y_m are floats in meters
   x_m = 1205.4
   y_m = 24.6

   state = CellStates.FUEL

   # Set cell which contains (x,y)'s state to FUEL
   fire.set_state_at_xy(x_m, y_m, state)

```

**Or by passing in the indices:**

```python
   from utilities.fire_util import CellStates

   # row and col must be integers
   row = 120
   col = 17


   state = CellStates.BURNT

   # Set cell at indices (row, col)'s state to BURNT
   fire.set_state_at_indices(row, col, state)

```

```{note}
While you can set a cell's state to FIRE using the above functions, it is recommended that you use the below functions to do so.
```

## Starting Fires
There are two sets of functions specific for setting fires within a cell. One set for starting wildfires, the other for setting prescribed fires. Each can be done in the same three ways states can be set:

**Passing in the cell object explicitly:**

```python
   # Set a wildfire at cell
   fire.set_wild_fire_at_cell(cell) # cell is an instance of 'Cell' class

   # Set a prescribed fire at cell
   fire.set_prescribed_fire_at_cell(cell) # cell is an instance of 'Cell' class

```

**Passing in the x,y coordinates:**

```python
   # x_m and y_m are floats in meters
   x_m = 1254.4
   y_m = 356.2

   # Set a wildfire at cell containing point (x,y)
   fire.set_wild_fire_at_xy(x_m, y_m)

   # Set a prescribed fire at cell containing point (x,y)
   fire.set_prescribed_fire_at_xy(x_m, y_m)

```

**Passing in the indices:**

```python
   
   # row and col must be integers
   row = 40
   col = 250

   # Set a wildfire at cell whose indices are (row, col)
   fire.set_wild_fire_at_indices(row, col)

   # Set a prescribed fire at cell whose indices are (row, col)
   fire.set_prescribed_fire_at_indices(row, col)

```

## Setting Fuel Content
The [fuel content](fire_modelling:fuel_content) of any cell in the sim can be set as well. The fuel content must be a float between 0-1, this represents the fraction of fuel remaining in a cell. This can be done any of the following ways:

**By passing in the cell object explicitly:**

```python

   # fuel_content must be between 0-1
   fuel_content = 0.4

   # Set the fuel content in cell to 0.4
   fire.set_fuel_content_at_cell(cell, fuel_content)

```

**By passing in the x,y coordinates:**

```python

   # fuel_content must be between 0-1
   fuel_content = 0.4

   # x_m and y_m are floats in meters
   x_m = 1254.4
   y_m = 356.2

   # Set the fuel content in cell which contains point (x,y)
   fire.set_fuel_content_at_xy(x_m, y_m, fuel_content)

```

**By passing in the indices:**

```python

   # fuel_content must be between 0-1
   fuel_content = 0.4

   # row and col must be integers
   row = 125
   col = 35

   # Set the fuel content in cell whose indices are (row, col)
   fire.set_fuel_content_at_indices(row, col, fuel_content)

```


## Setting Fuel Moisture
The [fuel moisture](fire_modelling:fuel_moisture) of a cell can be set as well. This sets the dead fuel moisture of the cell. Increasing the dead fuel moisture will slow the spread of fire, if the dead moisture is set at or above the fuel type's dead moisture of extinction, the likelihood that the cell will ignite approaches 0. Setting the fuel moisture is a good way to simulate the use of water or other fire suppressant to soak fuels. The fuel moisture can be set in the following ways:

**By passing in the cell object explicitly:**

```python
   # Set fuel moisture to 20%
   fuel_m = 0.2

   # Set fuel moisture at cell
   fire.set_fuel_moisture_at_cell(cell, fuel_m)

```

**By passing in the x,y coordinates:**

```python
   # Set fuel moisture to 20%
   fuel_m = 0.2

   # x_m and y_m are floats in meters
   x_m = 1254.4
   y_m = 356.2


   # Set fuel moisture at cell containing point (x,y)
   fire.set_fuel_moisture_at_xy(xy, fuel_m)

```

**By passing in the indices:**

```python
   # Set fuel moisture to 20%
   fuel_m = 0.2

   # row and col must be integers
   row = 125
   col = 356


   # Set fuel moisture at cell whose indices are (row, col)
   fire.set_fuel_moisture_at_indices(row, col, fuel_m)

```

## Get Wind Conditions
The current wind conditions can easily be accessed at anytime. Users can choose between two formats when retrieving wind conditions:


**Speed and Direction:**

Get the wind conditions broken up into wind speed and direction in m/s and degrees respectively.

```python
   # Get current wind conditions
   speed_m_s, dir_deg = fire.get_curr_wind_speed_dir()

```

**Velocity Component Vector:**

Get the wind conditions as an array of velocity components in m/s.

```python
   # Get current wind conditions
   wind_vec = fire.get_curr_wind_vec()

   # wind_vec contains the x and y components of the wind velocity
   x_vel_m_s, y_vel_m_s = wind_vec[0], wind_vec[1]

```
## Get Average Fire Coordinate
To find the average (x,y) position of all the cells on fire to estimate the center of the fire, the following function can be used:

```python

   x_avg, y_avg = fire.get_avg_fire_coord()

```

## Useful Properties
The pseudo-interface also provides read-only access to key properties of the simulation. Below are some of the key properties that can be accessed.

### cell_grid

The `cell_grid` property returns the raw backing array for the simulation

```python
   arr = fire.cell_grid
```

### cell_dict

The `cell_dict` property returns a dictionary of all the cell objects in the array, where the keys are the 'id' of each cell.

```python
   cell_dict = fire.cell_dict
```

### grid_height and grid_width

The `grid_height` property returns the max row of the sim's backing array, 'grid_width' returns the max column of the sim's backing array. 

```python
   max_row = fire.grid_height
   max_col = fire.grid_width
```

### x_lim and y_lim

The `x_lim` property returns the max x coordinate in the sim's map in meters, `y_lim` returns the max y coordinate in the sim's map in meters

```python
   max_x = fire.x_lim
   max_y = fire.y_lim
```

### curr_fires

The `curr_fires` property returns a set of Cell objects that are currently on fire.

```python
   fires = fire.curr_fires
```

### burnt_cells

The `burnt_cells` property  returns a set of Cell objects that are already burnt.

```python
   burnt_cells = fire.burnt_cells
```

### frontier

The `frontier` property returns a set of all the Cell objects in the 'fuel' state, that are also a neighbor to at least one cell that is on fire. These are the cells eligible to be ignited in the next time-sttep.

```python
   frontier = fire.frontier
```

### fire_breaks and fire_break_cells

The `fire_breaks` property returns a list of dictionaries representing the fire-breaks for the map. Each dictionary contains a representation of a LineString object and a fuel value ([see Map Files](map_creation:files)).

The `fire_break_cells` property returns a list of all the cells that are members of a fire-break within a sim's map.

```python
   fire_breaks = fire.fire_breaks

   fire_break_cells = fire.fire_break_cells
```

### roads

The `roads` property returns a list of (x,y) coordinates, each paired with a fuel content ((x,y), fuel content) representing the locations of all the points along a sim's roads and the fuel content used to model them.

```python
   roads = fire.roads

   (x, y), fuel_content = roads[0]

```

```{note}
The example functions and properties provided here are not comprehensize, see [base_classes.base_fire](./_autosummary/base_classes.base_fire.rst) for full documentation.
```





