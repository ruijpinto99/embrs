#include <unordered_map>
#include <tuple>
#include <map>

struct Position {
    float x, y, z;
};

enum ActionType {
	SET_MOISTURE = 0,
	SET_FUEL_CONTENT = 1,
	SET_PRESCIBED_BURN = 2,
	SET_WILDFIRE
};

struct Action {
	ActionType type;
	std::pair<float, float> pos;
	float time;
	float value; // TODO: Figure out how to specify fire type with float
};

struct ActionComparator {
	bool operator()(const Action* lhs, const Action* rhs) const {
		return lhs->time > rhs->time;
	}
};

struct CacheEntry {
	double prob = -1.0;
	double v_prop = -1.0;
	double alpha_h = -1.0;
	double k_phi = -1.0;
	double alpha_w = -1.0;
	double k_w = -1.0;
	double e_m = -1.0;
	double nc_factor = -1.0;
	double p_n = -1.0;
	double v_n = -1.0;
};

struct WindVec {
	double mag_m_s;
	double dir_rad;
};

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2;  // Combine the hash values
    }
};

struct TupleHash {
	template <typename T1, typename T2>
	std::size_t operator()(const std::tuple<T1, T2>& tuple) const {
		auto hash1 = std::hash<T1>{}(std::get<0>(tuple));
		auto hash2 = std::hash<T2>{}(std::get<1>(tuple));
		return hash1 ^ (hash2 << 1); // Hash function
	}
};

class HexUtil {
public:
	static inline const double hex_angle = 60 * (M_PI / 180.0);

	static inline const std::unordered_map<std::pair<int, int>, std::pair<double, double>, pair_hash> even_neighborhood_mapping = {
		{{-1, 1}, {-std::cos(M_PI / 3), std::sin(M_PI / 3)}},
		{{0, 1}, {std::cos(M_PI / 3), std::sin(M_PI / 3)}},
		{{1, 0}, {1, 0}},
		{{0, -1}, {std::cos(M_PI / 3), -std::sin(M_PI / 3)}},
		{{-1, -1}, {-std::cos(M_PI / 3), -std::sin(M_PI / 3)}},
		{{-1, 0}, {-1, 0}}
	};


	static inline const std::unordered_map<std::pair<int, int>, std::pair<double, double>, pair_hash> odd_neighborhood_mapping = {
		{{1, 0}, {1, 0}},
		{{1, 1}, {std::cos(M_PI / 3), std::sin(M_PI / 3)}},
		{{0, 1}, {std::cos(M_PI / 3), std::sin(M_PI / 3)}},
		{{-1, 0}, {-1, 0}},
		{{0, -1}, {-std::cos(M_PI / 3), -std::sin(M_PI / 3)}},
		{{1, -1}, {std::cos(M_PI / 3), -std::sin(M_PI / 3)}}
	};

	// Neighborhood definitions
    static inline const std::vector<std::pair<int, int>> even_neighborhood = {
		{-1, 1}, {0, 1}, {1, 0}, {0, -1}, {-1, -1}, {-1, 0}
	};
    
	static inline const std::vector<std::pair<int, int>> odd_neighborhood = {
		{1, 0}, {1, 1}, {0, 1}, {-1, 0}, {0, -1}, {1, -1}
	};

};


class WindAdjustments {
public:
	static inline const std::map<int, std::tuple<double, double, double>> wind_speed_param_mapping = {
        {0, {0.0, 1.0, 1.0}},
        {10, {0.33, 114.93, 0.89}},
        {20, {1.17, 84.83, 0.64}},
        {30, {2.12, 60.95, 0.4}},
        {50, {2.8, 45.13, 0.25}},
        {60, {3.03, 39.61, 0.2}},
        {70, {3.26, 33.92, 0.15}},
        {80, {3.49, 27.56, 0.1}},
        {90, {3.72, 20.0, 0.05}},
        {100, {3.94, 7.71, 0.01}}
    };
};

#pragma once