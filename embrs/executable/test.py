import subprocess
import numpy as np
import time
import struct
import mmap

def read_data(filename):
    total_ignitions = 0

    # Open the file and memory-map it
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Start reading from the beginning
        offset = 0
        while offset < mm.size():
            # Read an integer for the time step
            time_step = struct.unpack_from('f', mm, offset)[0]
            offset += struct.calcsize('f')

            # Read an integer for the number of cells
            num_cells = struct.unpack_from('i', mm, offset)[0]
            offset += struct.calcsize('i')

            # Print the time step and the number of cells
            # print(f"Time Step {time_step} has {num_cells} cells:")

            # Loop through the number of cells
            for _ in range(num_cells):
                # Each cell consists of two integers (x, y)
                x, y = struct.unpack_from('ff', mm, offset)
                offset += struct.calcsize('ff')

                # Print the cell coordinates
                # print(f"  Cell at ({x}, {y})")
                total_ignitions += 1
        # Close the mmap
        mm.close()

    print(f"total_ignitions: {total_ignitions}")

bias = 1
time_horizon_hr = 2
time_step = 15
cell_size = 15

rows = 100
cols = 100
# filename ='mmap.bin'

cell_type = np.dtype([
    ('id', np.int32),
	('state', np.int32),
    ('fuelType', np.int32),
	('fuelContent', np.float32),
	('moisture', np.float32),
	('position', [('x', np.float32), ('y', np.float32), ('z', np.float32)]),
    ('indices', [('i', np.int32), ('j', np.int32)]),
    ('changed', np.bool_)
])

# Create an array of cells
cells = np.zeros((rows, cols), dtype=cell_type)

wind_time_step_min = 15
wind_forecast = [[2, 2], [2, 3], [2, 4], [5, 5]]

wind_forecast_str = ' '.join(f"{u} {v}" for u, v in wind_forecast)

# TODO: Implement sending actions in format that c++ can read

# TODO: will need a way to generate this from fire state quickly (might have to convert everything to this)
# Initialize cells with some data


action_type = np.dtype([
    ('type', np.int32),
    ('pos', [('x', np.float32), ('y', np.float32)]),
    ('time', np.float32),
    ('value', np.float32)
])


action_time = 0.0
num_actions = 20

actions = np.zeros(num_actions, dtype=action_type)

for i in range(num_actions):
    actions[i]['type'] = 2 # TODO: need to formally define these types in python
    actions[i]['pos']['x'] = np.random.rand() * 15 * cols
    actions[i]['pos']['y'] = np.random.rand() * 15 * rows
    actions[i]['time'] = 0.0
    actions[i]['value'] = 0.2
    
action_filename = 'actions.dat'
file_size = actions.nbytes

# Ensure the file is sized correctly
with open(action_filename, 'wb') as f:
    f.seek(file_size - 1)
    f.write(b'\x00')

# Map the file and write data
with open(action_filename, 'r+b') as f:
    mm = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_WRITE)
    mm.write(actions.tobytes())
    mm.close()

# Initialize and set indices within the loop
id_ctr = 0
for i in range(rows):
    for j in range(cols):
        if np.random.rand() < 0.5:
            state = 2
        else:
            state = 1

        if i % 2 == 0:
            x = j * cell_size * np.sqrt(3)
        else:
            x = (j + 0.5) * cell_size * np.sqrt(3)

        y = i * cell_size * 1.5

        cells[i, j]['id'] = id_ctr
        cells[i, j]['state'] = state
        cells[i, j]['fuelType'] = 2
        cells[i, j]['fuelContent'] = 0.5
        cells[i, j]['moisture'] = 0.08
        cells[i, j]['position']['x'] = x
        cells[i, j]['position']['y'] = y
        cells[i, j]['position']['z'] = np.random.rand() * 15
        cells[i, j]['indices']['i'] = i  # Setting index i
        cells[i, j]['indices']['j'] = j
        cells[i, j]['changed'] = False

        id_ctr += 1
# Write the array to a memory-mapped file
cell_filename = 'cells.dat'
file_size = cells.nbytes

write_start = time.time()

# Ensure the file is sized correctly
with open(cell_filename, 'wb') as f:
    f.seek(file_size - 1)
    f.write(b'\x00')

# Map the file and write data
with open(cell_filename, 'r+b') as f:
    mm = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_WRITE)
    mm.write(cells.tobytes())
    mm.close()


# Pass in fire prediction settings
params = f"{num_actions} {bias} {time_horizon_hr} {time_step} {cell_size} {rows} {cols} {wind_time_step_min} {wind_forecast_str}"

result = subprocess.run(['./fire_prediction', cell_filename, action_filename], input=params, text=True, capture_output=True)

read_data('prediction.bin')

print(result.stdout)
write_end = time.time()

print(f"Entire process took: {write_end - write_start}")










