#include "fuel_consts.h"
#include <unordered_map>
#include <tuple>
#include "cell.h"
#include "util.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <random>
#include <cmath>
#include <algorithm>
#include <queue>

using namespace std;

// Inputs from python
	// (int []) cell states
	// (float []) fuel contents
	// (float) bias
	// (float) time_horizon_hr
	// (float) time_step_sec
	// (float) cell_size_m
	// (float []) wind_forecast (add noise before inputting)
	// (float) wind_forecast_t_step
	// action sets - need to figure out best way to format this

// Output
	// predicted fires - need a good way to format this

// Declare global variable
using IgnPoint = std::pair<float, float>;
using PredMap = std::map<float, std::vector<IgnPoint>>;

PredMap prediction;
std::vector<std::vector<Cell*>> cellGrid; // (rows, std::vector<Cell*>(cols));
std::priority_queue<Action*, std::vector<Action*>, ActionComparator> action_queue;
std::unordered_map<std::tuple<int,int>, CacheEntry, TupleHash> relational_dict;
std::vector<WindVec> wind_forecast;
WindVec curr_wind;
bool wind_changed = false;
double bias, horizon, t_step, c_size, wind_forecast_t_step;
int rows, cols, num_actions;
double sim_time = 0;
int curr_wind_index = -1;

// Start General Fire Sim Functions // TODO: Figure out best design for these functions to live elsewhere

std::pair<int, int> hex_round(double q, double r) {
	// Perform hexagonal indices rounding
	
	double s = -q - r;
	double q_r = std::round(q);
	double r_r = std::round(r);
	double s_r = std::round(s);

	double q_diff = std::abs(q_r - q);
	double r_diff = std::abs(r_r - r);
	double s_diff = std::abs(s_r - s);


	if (q_diff > r_diff && q_diff > s_diff) {
		q_r = -r_r - s_r;
	} else if (r_diff > s_diff) {
		r_r = -q_r - s_r;
	} else {
		s_r = -q_r - r_r;
	}

	return {static_cast<int>(q_r), static_cast<int>(r_r)};
}

Cell* get_cell_from_xy(std::pair<float, float> xy_pos) {
	// Get cell which contains a given (x, y) point
	float x_m = xy_pos.first;
	float y_m = xy_pos.second;

	// Convert to hexagonal coordinates
	double q = (std::sqrt(3) / 3.0 * x_m - 1.0/3.0 * y_m) / c_size;
	double r = (2.0/3.0 * y_m) / c_size;

	// Round coordinates to ints
	std::pair<int, int> qr = hex_round(q, r);

	// Convert to rectangular array indices
	int row = qr.second;
	int col = qr.first + row / 2.0;

	// Return cell at calculated indices
	return cellGrid[row][col];
}

void perform_action(Action* action, std::vector<Cell*> curr_fires) {
	// Carry out specific action specified

	// Get cell at the specified action location
	std::pair<float, float> xy_pos = action->pos;
	Cell* cell = get_cell_from_xy(xy_pos);

	switch (action->type) {
	
	case SET_MOISTURE:
		cell->moisture = action->value;
		break;
	
	case SET_FUEL_CONTENT:
		cell->fuelContent = action->value;
		break;
	
	case SET_PRESCIBED_BURN:
		cell->state = PRESCRIBED_FIRE;
		curr_fires.push_back(cell); 
		break;

	case SET_WILDFIRE:
		cell->state = WILDFIRE;
		curr_fires.push_back(cell);
	
	default:
		break;
	}
}

void perform_actions(std::vector<Cell*> curr_fires) {
	// Perform the actions scheduled before the next sim time-step
	while (!action_queue.empty() && sim_time >= action_queue.top()->time) {
		Action* action = action_queue.top();
		perform_action(action, curr_fires);
		action_queue.pop();
	}
}

std::pair<double, double> calc_slope_effect(Cell* curr_cell, Cell* neighbor) {
	// Calculate the effect of the slope between curr_cell and neighbor

	// Get relevant distances between cell positions
	double rise = neighbor->position.z - curr_cell->position.z;
	double dx = neighbor->position.x - curr_cell->position.x;
	double dy = neighbor->position.y - curr_cell->position.y;
	double run = std::sqrt(std::pow(dx, 2) + std::pow(dy, 2));

	// Calculate slope percentage
	double slope_pct = (rise / run) * 100;
	
	// Calculate effect on probability
	double alpha_h;
	if (slope_pct == 0) {
		alpha_h = 1;
	} else if (slope_pct < 0) {
		alpha_h = 0.5/(1 + std::exp(-0.2*(slope_pct + 40))) + 0.5;
	} else {
		alpha_h = 1/(1 + std::exp(-0.2*(slope_pct - 40))) + 1;
	}

	// Get slope angle
	double phi = std::atan(rise/run);

	// Calculate effect on propagation velocity
	int A;
	if (phi < 0) {
		A = 1;
		phi = -phi;
	} else {
		A = 0;
	}

	double k_phi = std::exp(std::pow(-1,A)* 3.533 * std::pow(tan(phi), 1.2));

	return {alpha_h, k_phi};
}

double norm(const std::pair<double, double>& v) {
	// Calculate the norm of a pair
	double sum = v.first * v.first + v.second * v.second;
    return std::sqrt(sum);
}

double lorentzian(double x, double A, double gamma, double C) {
	// Sample the lorentzian function at x using the given parameters
	return A / (1 + std::pow((x/gamma), 2)) + C;
}

double interpolate_wind_adjustment(double wind_speed_kmh, double direction) {
	// Interpolate the Lorentzian parameters to use based on wind speed and direciton

	// TODO: should further verify that this works

	std::map<int, std::tuple<double, double, double>> param_mapping = WindAdjustments::wind_speed_param_mapping;
	
	// Check if the exact wind speed exists
    auto it = param_mapping.find(static_cast<int>(wind_speed_kmh));
    if (it != param_mapping.end()) {

		auto [A, gamma, C] = it->second;

        return lorentzian(wind_speed_kmh, A, gamma, C);
    }

    // Find the closest lower and upper wind speeds
    auto lower_it = param_mapping.lower_bound(static_cast<int>(wind_speed_kmh)); // Returns iterator to first element not less
    auto upper_it = lower_it;

    if (lower_it == param_mapping.begin()) {
        upper_it++;
    } else if (lower_it == param_mapping.end()) {
        lower_it--;
    } else {
        lower_it--;
    }

    // Extract parameters
    int v_lower = lower_it->first;
    int v_upper = upper_it->first;
    std::tuple<double, double, double> lower_params = lower_it->second;
    std::tuple<double, double, double> upper_params = upper_it->second;

    // Calculate weights for interpolation
    double w1 = (static_cast<double>(v_upper) - wind_speed_kmh) / (v_upper - v_lower);
    double w2 = 1 - w1;

    // Interpolate the parameters
    double A_lower, gamma_lower, C_lower;
    double A_upper, gamma_upper, C_upper;

    std::tie(A_lower, gamma_lower, C_lower) = lower_params;
    std::tie(A_upper, gamma_upper, C_upper) = upper_params;

    double A_interp = w1 * A_lower + w2 * A_upper;
    double gamma_interp = w1 * gamma_lower + w2 * gamma_upper;
    double C_interp = w1 * C_lower + w2 * C_upper;

    return lorentzian(wind_speed_kmh, A_interp, gamma_interp, C_interp);
}

std::pair<double, double> calc_wind_effect(Cell* curr_cell, Cell* neighbor) {
	// Calculate wind effect

	// Get the difference in indices between cells to find neighborhood mapping
	int di = neighbor->indices.first - curr_cell->indices.first; //TODO: check that these are correctly ordered
	int dj = neighbor->indices.second - curr_cell->indices.second;

	// Determine correct neighborhood mapping to use
	std::unordered_map<std::pair<int, int>, std::pair<double, double>, pair_hash> mapping;
	if (curr_cell->indices.first % 2 == 0){
		mapping = HexUtil::even_neighborhood_mapping;
	} else {
		mapping = HexUtil::odd_neighborhood_mapping;
	}
	
	// Get the unit vector pointing from curr_cell to neighbor
	std::pair<double, double> disp_vec = mapping.at({di, dj});

	// Calculate the current wind vector // TODO: may be more efficient to calculate this just once
	std::pair<double, double> vec = {curr_wind.mag_m_s * std::cos(curr_wind.dir_rad), curr_wind.mag_m_s * std::sin(curr_wind.dir_rad)};

	// Calculate dot projection between unit vector and wind vector
	double dot_proj = disp_vec.first * vec.first + disp_vec.second * vec.second;
	
	// Calculate velocity adjustmet
	double k_w = std::max(0.0, std::exp(0.1783 * dot_proj) - 0.486);

	// Calculate probability adjustment
	double alpha_w;
	if (curr_wind.mag_m_s == 0) {
		alpha_w = 1;
	} else {
		double cos_rel_angle = dot_proj / (norm(disp_vec) * norm(vec));
		double rel_angle = std::acos(cos_rel_angle);
		rel_angle = rel_angle * (180.0 / M_PI);

		double adj_vel_kmh = std::max(curr_wind.mag_m_s * 3.6 - 8, 0.0);
		alpha_w = interpolate_wind_adjustment(adj_vel_kmh, rel_angle);

	}

	return {alpha_w, k_w};
}

double calc_fuel_moisture_effect(double fm_ratio) {
	// Calculate effect of fuel moisture
	fm_ratio = std::min(fm_ratio, 1.0);
	double e_m = -4.5*std::pow(fm_ratio - 0.5, 3) + 0.5625;

	return e_m;
}

std::pair<double, double> calc_prob(Cell* curr_cell, Cell* neighbor) {
	// Calculate the probability of curr_cell igniting neighbor

	// Generate corresponding key and check if a map entry exists
	std::tuple<int, int> key{curr_cell->id, neighbor->id};
	auto it = relational_dict.find(key);

	// If there is an entry and nothing relevant has changed, return last calculated result
	if (it != relational_dict.end() && !(wind_changed || neighbor->changed || curr_cell->changed) ) {
		return {it->second.prob, it->second.v_prop};
	} 

	// Get corresponding entry or create a new one
	CacheEntry curr_entry;
	if (it != relational_dict.end()) {
		curr_entry = it->second;
	} else {
		// Calculate effect of slope between cells
		std::pair<double, double>  slope_effects = calc_slope_effect(curr_cell, neighbor);

		curr_entry.alpha_h = slope_effects.first;
		curr_entry.k_phi = slope_effects.second;
	}

	if (wind_changed || curr_entry.alpha_w == -1.0){
		// Calculate effect of the wind
		std::pair<double, double> wind_effects = calc_wind_effect(curr_cell, neighbor);
		
		curr_entry.alpha_w = wind_effects.first;
		curr_entry.k_w = wind_effects.second;
	}

	if (neighbor->changed || curr_entry.e_m == -1.0) {
		// Calculate effect of the moisture of the neighbor cell
		float dead_m_ext = FuelConstants::dead_fuel_moisture_ext_table.at(neighbor->fuelType);
		float dead_m = neighbor->moisture;
		float fm_ratio = dead_m/dead_m_ext;

		curr_entry.e_m = calc_fuel_moisture_effect(fm_ratio);
		curr_entry.nc_factor = neighbor->fuelContent;

		neighbor->changed = true;
	}

	if (curr_cell->changed || curr_entry.p_n == -1.0) {

		// Get the nominal probability and propagation velocity for the current cell fuel type
		curr_entry.p_n = FuelConstants::nom_spread_prob_table.at(curr_cell->fuelType).at(neighbor->fuelType);
		curr_entry.v_n = FuelConstants::nom_spread_vel_table.at(curr_cell->fuelType);

		if (curr_cell->state == PRESCRIBED_FIRE) {
			curr_entry.p_n *= ControlledBurnParams::nominal_prob_adj;
			curr_entry.v_n *= ControlledBurnParams::nominal_vel_adj / 60;
		}
		
		curr_cell->changed = false;
	}

	// Combine wind and slope effects
	double alpha_wh = curr_entry.alpha_w * curr_entry.alpha_h;
	
	// Calculate propagation velocity
	curr_entry.v_prop = curr_entry.v_n * curr_entry.k_w * curr_entry.k_phi;

	// Calculate predicted spread time between cells
	double delta_t_sec;
	if (curr_entry.v_prop == 0.0) {
		delta_t_sec = std::numeric_limits<double>::infinity();

	} else {
		delta_t_sec = (c_size * 1.5) / curr_entry.v_prop;
	}

	// Calculate the nominal number of iterations
	double num_iters = delta_t_sec/t_step;

	// Calculate probability
	double prob = std::pow(1-(1-curr_entry.p_n), alpha_wh) * curr_entry.e_m;
	prob = 1 - std::pow((1-prob), (1/num_iters));
	prob *= curr_entry.nc_factor;
	curr_entry.prob = prob;

	return {curr_entry.prob, curr_entry.v_prop};
}

std::vector<Cell*> get_neighbors(int i, int j) {
    // Select the appropriate neighborhood based on the row index
    const std::vector<std::pair<int, int>>& neighborhood = (i % 2 == 0) ? HexUtil::even_neighborhood : HexUtil::odd_neighborhood;

	// Populate neighbors vector
    std::vector<Cell*> neighbors;
    for (const auto& offset : neighborhood) {
        int ni = i + offset.first;
        int nj = j + offset.second;

        // Check boundaries
        if (ni >= 0 && ni < cellGrid.size() && nj >= 0 && nj < cellGrid[ni].size()) {
            neighbors.push_back(cellGrid[ni][nj]);
        }
    }

    return neighbors;
}

bool update_wind(double sim_time) {
	// Update the current wind vec based on current time step and the forecast

	double curr_time_min = sim_time / 60;
	int curr_index = static_cast<int>(std::floor(curr_time_min / wind_forecast_t_step));

	if (curr_index != curr_wind_index) {
		curr_wind_index = curr_index;

		curr_wind = wind_forecast[curr_wind_index];
		
		return true;
	}
	return false;
}

// Start Prediction Specific Functions

void add_cell_to_pred(PredMap& pred, float time_sec, IgnPoint point) {
	// Add a cell to the prediction at the time step when ignition has been predicted
	pred[time_sec].push_back(point);
}

void iterate(std::vector<Cell*>& curr_fires, std::default_random_engine& engine, std::uniform_real_distribution<double>& dist) {
	// Run a single time-step of prediction model

	// Update wind vector
	wind_changed = update_wind(sim_time);

	// Perform actions if any remain in the queue
	if (!action_queue.empty()) {
		perform_actions(curr_fires);
	}

	std::vector<Cell*> new_fires;

	// Loop through current fires
	auto it = std::remove_if(curr_fires.begin(), curr_fires.end(),
	[&](Cell* cell) {
		// Get indices of current cell
		std::pair<int, int> indices = cell->indices;
		// Get neighbors of current cell // TODO: doing this every iteration is very inefficient
		std::vector<Cell*> neighbors = get_neighbors(indices.first, indices.second);
		bool has_burnable_neighbors = false;

		// Loop through neighbors of fires
		for (Cell* neighbor : neighbors) {
			if (neighbor->state == FUEL || 
				cell->state == WILDFIRE && neighbor->state == PRESCRIBED_FIRE ||
				cell->state == PRESCRIBED_FIRE && neighbor->fuelContent > ControlledBurnParams::min_burnable_fuel_content) {
				
				has_burnable_neighbors = true;

				// Calculate ignition prob
				std::pair<float, float> prob_output = calc_prob(cell, neighbor);

				float prob = prob_output.first * bias;
				float v_prop = prob_output.second * bias;

				// Generate random number and make ignition decision
				double rand = dist(engine);
				if (rand < prob) {
					
					// Set neighbors state to fire
					neighbor->state = cell->state;
					neighbor->changed = true;

					// Add neighbor to new fires
					new_fires.push_back(neighbor); 

					// Add to predicted ignitions output
					add_cell_to_pred(prediction, sim_time, {neighbor->position.x, neighbor->position.y});
				}
			}

			if (cell->state == PRESCRIBED_FIRE) {
				cell->fuelContent *= 0.3;

				// TODO: add to partially burned output
			}
		}

		return !has_burnable_neighbors; // Return true if cell should be removed
	});

	// Erase cells without burnable neighbors from curr_fires
	curr_fires.erase(it, curr_fires.end());

	// Add new fires to curr_fires
    curr_fires.insert(curr_fires.end(), new_fires.begin(), new_fires.end());
}

void writePredToFile(const PredMap& prediction){
	// Write prediction map to a .bin file for python to processs
    std::ofstream file("prediction.bin", std::ios::binary);
    for (const auto& entry : prediction) {
        float timeStep = entry.first;
        int numCells = entry.second.size();

        file.write(reinterpret_cast<const char*>(&timeStep), sizeof(timeStep));
        file.write(reinterpret_cast<const char*>(&numCells), sizeof(numCells));
        for (const auto& cell : entry.second) {
            file.write(reinterpret_cast<const char*>(&cell.first), sizeof(cell.first));
            file.write(reinterpret_cast<const char*>(&cell.second), sizeof(cell.second));
        }
    }

    file.close();
}

int main(int argc, char* argv[]) {
	// Parse Inputs
	if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << "<cell_filename>" << "<action_filename>" << std::endl;
        return 1;
    }

    const char* cell_filename = argv[1];
	const char* action_filename = argv[2];
 

	// Get sim parameters from input string
	string line;
	getline(cin, line);
	stringstream ss(line);
	ss >> num_actions >>bias >> horizon >> t_step >> c_size >> rows >> cols >> wind_forecast_t_step;
	
	// Construct wind forecast
	WindVec wv;
	while (ss >> wv.mag_m_s >> wv.dir_rad) {
		wind_forecast.push_back(wv);
	}

	// Get cells from input file
	size_t size = rows*cols*sizeof(Cell);
    int fd = open(cell_filename, O_RDWR);

	if (fd == -1) {
		std::cout << "Failed to open the file: " << cell_filename << std::endl;
		return 1;
	}

	ftruncate(fd, size);
	Cell* cells = static_cast<Cell*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

	if (cells == MAP_FAILED) {
		close(fd);
		perror("Error mapping the file");
		std::cout << "Error mapping the file";
		return 1;
	}

	// Format array of cells in a 2d grid analogous to numpy array
	cellGrid.resize(rows, std::vector<Cell*>(cols, nullptr));
	for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cellGrid[i][j] = cells + i * cols + j;
        }
    }

    // Get cells that start initially on fire
    std::vector<Cell*> curr_fires;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (cellGrid[i][j]->state == 2) {
                curr_fires.push_back(cellGrid[i][j]);
            }
        }
    }

    close(fd);


	// Load actions from input file
	size = num_actions * sizeof(Action);
	fd = open(action_filename, O_RDWR);

	if (fd == -1) {
		std::cout << "Failed to open the file: " << action_filename << std::endl;
		return 1;
	}

	ftruncate(fd, size);

	Action* actions = static_cast<Action*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

	if (actions == MAP_FAILED) {
		close(fd);
		perror("Error mapping the file");
		std::cout << "Error mapping the file";
		return 1;
	}

	// Add actions to a priority queue according to time
	for (int i = 0; i < num_actions; i++) {
		action_queue.push(actions + i);
	}

	// Create a random engine
    std::random_device rd;
    std::default_random_engine engine(rd());

    // Create a distribution from [0, 1) for random number generation
    std::uniform_real_distribution<double> dist(0.0, 1.0);

	// Start sim loop
	bool done = false;
	while (!done) {   
		iterate(curr_fires, engine, dist);
		sim_time += t_step;
		done = curr_fires.empty() || sim_time >= (horizon * 3600);
	}

	// Write results to memory map
	writePredToFile(prediction);

	return 0;
}