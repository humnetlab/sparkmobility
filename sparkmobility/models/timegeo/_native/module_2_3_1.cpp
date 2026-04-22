// Compile with: (extra libraries to be specified: apache arrow, h3, snappy)
// g++ -std=c++17 -O3 -fopenmp -o module_2_3_1 module_2_3_1.cpp \
//   -I$HOME/local/include \
//   -I$HOME/apache-arrow-20.0.0/cpp/src \
//   -I$HOME/apache-arrow-20.0.0/cpp/build/src \
//   -I$HOME/h3/build/src/h3lib/include \
//   -I$HOME/anaconda3/include/python3.11 \
//   -I$HOME/anaconda3/lib/python3.11/site-packages/pybind11/include \
//   -L$HOME/local/lib \
//   -L$HOME/apache-arrow-20.0.0/cpp/build/release \
//   -L$HOME/h3/build/lib \
//   -larrow -lparquet -lh3 -lsnappy \
//   $HOME/anaconda3/lib/libpython3.11.so \
//   -lstdc++fs -pthread



// g++ -std=c++17 -O3 -fopenmp -o module_2_3_1 module_2_3_1.cpp \
//   -I$HOME/local/include \
//   -I$HOME/apache-arrow-20.0.0/cpp/src \
//   -I$HOME/apache-arrow-20.0.0/cpp/build/src \
//   -I$HOME/h3/build/src/h3lib/include \
//   -I$HOME/anaconda3/envs/sparkmobility-testing/include/python3.11 \
//   -I$HOME/anaconda3/envs/sparkmobility-testing/lib/python3.11/site-packages/pybind11/include \
//   -L$HOME/apache-arrow-20.0.0/cpp/build/release \
//   -L$HOME/local/lib \
//   -L$HOME/h3/build/lib \
//   -Wl,-rpath,$HOME/apache-arrow-20.0.0/cpp/build/release \
//   -larrow -lparquet -lh3 -lsnappy \
//   $HOME/anaconda3/envs/sparkmobility-testing/lib/libpython3.11.so \
//   -lstdc++fs -pthread

// Command for running the binary
// ./module_2_3_1 /data_1/aparimit/imelda_data/outputs_imelda/2019112900/2019112900_1/work_locations.parquet ./results/Parameters 0 2 3000 1.0 600 0.6 -0.21 2>&1 | tee run.log

#include <h3/h3api.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <utility>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <unordered_map>
#include <map>
#include <memory>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/resource.h>
#include <set>
#include <unordered_set>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;

#define NaNUM 8
#define LaNUM 10
#define NBINS 10
#define secOneDay 86400
#define winterTimeStart 1667552400
#define CHUNK_SIZE 10000

using namespace std;

// Suppress per-row-group progress chatter when --quiet is passed (or
// TIMEGEO_QUIET=1 is set in the environment). All other output still emits
// so the Python-side log filter can pick up genuinely useful lines.
static bool g_quiet = false;

// Adapted from 2-DT_Comm.cpp
vector<double> real_time;
vector<double> real_dt;
vector<int> real_loc;
vector<int> real_day;
vector<int> real_locid;
vector<double> real_lon;
vector<double> real_lat;

vector<double> simu_time;
vector<double> simu_dt;
vector<int> simu_loc;
vector<int> simu_day;
vector<int> simu_locid;
vector<double> simu_lon;
vector<double> simu_lat;

vector<double> best_real_time;
vector<double> best_real_dt;
vector<int> best_real_loc;
vector<int> best_real_day;
vector<int> best_real_locid;
vector<double> best_real_lon;
vector<double> best_real_lat;

vector<double> best_simu_time;
vector<double> best_simu_dt;
vector<int> best_simu_loc;
vector<int> best_simu_day;
vector<int> best_simu_locid;
vector<double> best_simu_lon;
vector<double> best_simu_lat;

std::vector<double> daily_activeness;
std::vector<double> weekly_activeness;
std::vector<double> daily_weekly_activeness;

std::vector<double> original_real_dt;
std::vector<double> rescaled_real_dt;
std::vector<double> original_simu_dt;
std::vector<double> rescaled_simu_dt;

const int rand_range = 600; // or whatever value is appropriate

// Define the StayRegion struct
struct StayRegion {
    std::string user_id;
    int64_t timestamp;
    int location_type = 0; // 1 = home, 2 = work, 0 = other
    int day_of_week = 0;
    std::string work_h3_index;
    std::string home_h3_index;
    double lat = 0.0;
    double lon = 0.0;
    int location_id = 0;
    double home_lat = 0.0;
    double home_lon = 0.0;
    double work_lat = 0.0;
    double work_lon = 0.0;
};

// Set memory limit for the process
void set_memory_limit(long memory_mb) {
    struct rlimit rl;
    rl.rlim_cur = memory_mb * 1024 * 1024;
    rl.rlim_max = memory_mb * 1024 * 1024;

    if (setrlimit(RLIMIT_AS, &rl) == -1) {
        std::cerr << "Error setting memory limit" << std::endl;
    } else {
        std::cout << "Memory limit set to " << memory_mb << " MB" << std::endl;
    }
}

// Get all Parquet files from a directory
std::vector<std::string> get_parquet_files(const std::string& path) {
    std::vector<std::string> parquet_files;

    // Check if path is a directory
    struct stat path_stat;
    stat(path.c_str(), &path_stat);

    if (S_ISDIR(path_stat.st_mode)) {
        DIR *dir = opendir(path.c_str());
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                std::string file_name = entry->d_name;

                // Check if file has .parquet extension and doesn't contain .crc
                if ((file_name.find(".parquet") != std::string::npos) &&
                    (file_name.find(".crc") == std::string::npos)) {
                    parquet_files.push_back(path + "/" + file_name);
                }
            }
            closedir(dir);
        }
    } else if (path.find(".parquet") != std::string::npos) {
        // Single file
        parquet_files.push_back(path);
    }

    std::cout << "Found " << parquet_files.size() << " valid Parquet files in directory" << std::endl;
    return parquet_files;
}

// Function to clamp a value between a range
double clamp(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(value, max_value));
}

// Detect periodicity in a time series
void DetectPeriodicity(vector<double>& ts, vector<double>& periods, vector<double>& magnitudes) {
    int n = ts.size();
    if (n < 2) return;

    // Simple autocorrelation-based periodicity detection
    for (int lag = 1; lag < n / 2; lag++) {
        double sum = 0;
        int count = 0;

        for (int i = 0; i < n - lag; i++) {
            sum += fabs(ts[i] - ts[i + lag]);
            count++;
        }

        double avg_diff = sum / count;
        if (avg_diff < 0.1 * (*std::max_element(ts.begin(), ts.end()) -
                              *std::min_element(ts.begin(), ts.end()))) {
            periods.push_back(lag);
            magnitudes.push_back(1.0 / (avg_diff + 1e-6));
        }
    }
}

// Rescale time values
void RescaleTime(vector<double>& times, bool is_real) {
    if (times.size() <= 1) return;

    // Sort the times to ensure chronological order
    vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    for (int i = 0; i < sorted_times.size() - 1; i++) {
        double dt = sorted_times[i + 1] - sorted_times[i];
        // Only add positive time differences
        if (dt > 0) {
            if (is_real) {
                original_real_dt.push_back(dt);
                // Convert seconds to rounded integer hours, matching the original format
                rescaled_real_dt.push_back(round(dt / 3600));
            } else {
                original_simu_dt.push_back(dt);
                // Convert seconds to rounded integer hours, matching the original format
                rescaled_simu_dt.push_back(round(dt / 3600));
            }
        }
    }
}

// Function to correct timestamp to LA time
int64_t timeCorrect(int64_t timestamp) {
    int64_t timeOff = 0;
    if (timestamp < winterTimeStart) {
        timeOff = 4*3600;
    } else {
        timeOff = 5*3600;
    }
    return timestamp - timeOff;
}

// Rename to avoid conflict with H3 library
double convertRadsToDegrees(double rads) {
    return rads * 180.0 / M_PI;
}

// Add this function declaration before process_single_user
double AreaTestStat(const std::vector<double>& times, const std::vector<double>& simulated_time, int nbins);

// Process a single user's stay regions
void process_single_user(const std::vector<StayRegion>& regions,
                        FILE* fout_id1, FILE* fout_id2, FILE* fout_id3,
                        FILE* fout_id4, FILE* fout_id5, FILE* fout_id6, FILE* fout_id7,
                        int slot_interval) {
    if (regions.empty()) return;

    // Clear previous user data
    real_time.clear();
    real_dt.clear();
    real_loc.clear();
    real_day.clear();
    real_locid.clear();
    real_lon.clear();
    real_lat.clear();
    simu_time.clear();
    simu_dt.clear();
    simu_loc.clear();
    simu_day.clear();
    simu_locid.clear();
    simu_lon.clear();
    simu_lat.clear();

    // Extract stay times and generate time differences
    std::vector<StayRegion> sorted_regions = regions;
    std::sort(sorted_regions.begin(), sorted_regions.end(),
             [](const StayRegion& a, const StayRegion& b) {
                 return a.timestamp < b.timestamp;
             });

    // Home and work locations
    std::string home_h3 = "";
    std::string work_h3 = "";
    bool has_home = false;
    double home_lat = 0.0, home_lon = 0.0, work_lat = 0.0, work_lon = 0.0;
    bool has_work = false;
    double best_nw = 0.0;

    for (const auto& region : sorted_regions) {
        if (!region.home_h3_index.empty()) {
            home_h3 = region.home_h3_index;
            has_home = true;

            // Convert home H3 to lat/lon using cellToLatLng
            H3Index h3Index;
            if (stringToH3(home_h3.c_str(), &h3Index) == 0) {
                LatLng latLng;
                if (cellToLatLng(h3Index, &latLng) == 0) {
                    // Use the H3-converted coordinates for home coordinates
                    home_lat = convertRadsToDegrees(latLng.lat);
                    home_lon = convertRadsToDegrees(latLng.lng);
                    real_lat.push_back(home_lat);
                    real_lon.push_back(home_lon);
                }
            }
        }
        if (!region.work_h3_index.empty()) {
            work_h3 = region.work_h3_index;
            has_work = true;

            // Convert work H3 to lat/lon using cellToLatLng
            H3Index h3Index;
            if (stringToH3(work_h3.c_str(), &h3Index) == 0) {
                LatLng latLng;
                if (cellToLatLng(h3Index, &latLng) == 0) {
                    // Use the H3-converted coordinates for work coordinates
                    work_lat = convertRadsToDegrees(latLng.lat);
                    work_lon = convertRadsToDegrees(latLng.lng);
                    real_lat.push_back(work_lat);
                    real_lon.push_back(work_lon);
                }
            }
        }
    }

    // If either home or work location is missing, skip this user
    if (!has_home || !has_work) {
        return;
    }

    // Extract real trajectory
    for (size_t i = 0; i < sorted_regions.size(); i++) {
        real_time.push_back(sorted_regions[i].timestamp);
        real_loc.push_back(sorted_regions[i].location_type);
        real_day.push_back(sorted_regions[i].day_of_week);
        real_locid.push_back(sorted_regions[i].location_id);
        real_lon.push_back(sorted_regions[i].lon);
        real_lat.push_back(sorted_regions[i].lat);

        if (i > 0) {
            // Calculate time difference in hours
            double dt = (sorted_regions[i].timestamp - sorted_regions[i-1].timestamp) / 3600.0;
            real_dt.push_back(dt);
        }

        // Add this new block here
        // Update activity patterns for non-work locations
        if (sorted_regions[i].location_type != 2) {  // Not work location
            int daily_slot_num = secOneDay / slot_interval;
            daily_activeness.assign(daily_slot_num, 0.0);
            weekly_activeness.assign(7, 0.0);
            daily_weekly_activeness.assign(7 * daily_slot_num, 0.0);
            int curr_daily_slot = (int)((timeCorrect(sorted_regions[i].timestamp)) % secOneDay / slot_interval);
            int curr_weekly_slot = ((int)((timeCorrect(sorted_regions[i].timestamp)) / secOneDay)) % 7;

            daily_activeness[curr_daily_slot] += 1.0;
            weekly_activeness[curr_weekly_slot] += 1.0;
            daily_weekly_activeness[curr_weekly_slot * daily_slot_num + curr_daily_slot] += 1.0;
        }

    }

    // Proper TimeGeo model implementation - replace the simplified trajectory generation code
    // Define parameter arrays from original code
    double n1_arr[] = {16, 17, 18, 19, 20};
    int n1_num = 5;
    double n2_arr[] = {1, 2, 3, 4, 5};
    int n2_num = 5;

    // Initialize tracking variables
    double best_ats = std::numeric_limits<double>::max();
    int best_index1 = 0;
    int best_index2 = 0;

    // Optimization loop - try all parameter combinations
    for (int index1 = 0; index1 < n1_num; index1++) {
        double n1 = n1_arr[index1];

        for (int index2 = 0; index2 < n2_num; index2++) {
            double n2 = n2_arr[index2];

            // Run simulation with these parameters
            std::vector<double> simu_time_candidate;
            std::vector<int> simu_loc_candidate;
            double nw = 0.0;
            int day_end_at_home = 0;
            int day_not_end_at_home = 0;

            // Set simulation start time
            double current_time = real_time[0];
            double end_time = real_time[real_time.size()-1];

            // Initialize simulation
            bool at_home = true;
            bool at_work = false;
            simu_time_candidate.push_back(current_time);
            simu_loc_candidate.push_back(1); // Start at home

            // Calculate daily and weekly activeness based on parameters
            int daily_slot_num = secOneDay / slot_interval;
            std::vector<double> daily_activeness(daily_slot_num, 0.0);

            for (int i = 0; i < daily_slot_num; i++) {
                daily_activeness[i] = std::exp(-n1 * std::pow(std::abs(i - daily_slot_num/2) / (double)daily_slot_num, n2));
            }

            std::vector<double> weekly_activeness(7, 1.0); // Uniform for now

            // Main simulation loop
            while (current_time < end_time) {
                // Calculate next movement time
                double time_increment = slot_interval + (rand() % rand_range - rand_range/2);
                current_time += time_increment;

                if (current_time >= end_time) break;

                // Get current time slot
                int daily_slot = (int)((timeCorrect(current_time)) % secOneDay / slot_interval);
                int weekly_slot = (int)((timeCorrect(current_time)) / secOneDay) % 7;

                // End of day tracking
                if (daily_slot == daily_slot_num-1) {
                    if (at_home) {
                        day_end_at_home++;
                    } else {
                        day_not_end_at_home++;
                    }
                }

                if (at_home) {
                    // At home - decide whether to leave
                    double pt = daily_activeness[daily_slot] * weekly_activeness[weekly_slot];
                    if ((rand() % 100000) / 100000.0 < pt) {
                        // Leave home
                        at_home = false;

                        // Determine if going to work or other location
                        if (has_work && (rand() % 100) / 100.0 < 0.6) { // 60% chance to go to work if work exists
                            at_work = true;
                            simu_time_candidate.push_back(current_time);
                            simu_loc_candidate.push_back(2); // Work
                            nw += 1.0; // Count work visit
                        } else {
                            simu_time_candidate.push_back(current_time);
                            simu_loc_candidate.push_back(3); // Other location
                        }
                    }
                } else {
                    // Not at home - decide whether to move
                    double p_move;
                    if (at_work) {
                        // At work - probability based on work hours
                        int hour = (daily_slot * slot_interval) / 3600;
                        p_move = (hour < 9 || hour > 17) ? 0.3 : 0.05; // Higher probability to leave outside work hours
                    } else {
                        // At other location - base probability
                        p_move = 0.3;
                    }

                    if ((rand() % 100000) / 100000.0 < p_move) {
                        // Move somewhere
                        double p_home = 0.7; // 70% chance to go home
                        if ((rand() % 100000) / 100000.0 < p_home) {
                            // Go home
                            at_home = true;
                            at_work = false;
                            simu_time_candidate.push_back(current_time);
                            simu_loc_candidate.push_back(1); // Home
                        } else if (has_work && !at_work && (rand() % 100) / 100.0 < 0.4) {
                            // Go to work
                            at_work = true;
                            at_home = false;
                            simu_time_candidate.push_back(current_time);
                            simu_loc_candidate.push_back(2); // Work
                            nw += 1.0; // Count work visit
                        } else {
                            // Go to another other location
                            at_work = false;
                            simu_time_candidate.push_back(current_time);
                            simu_loc_candidate.push_back(3); // Other location
                        }
                    }
                }
            }

            // Normalize nw by the number of days in simulation
            if (!simu_time_candidate.empty()) {
                double simulation_days = (simu_time_candidate.back() - simu_time_candidate.front()) / secOneDay;
                if (simulation_days > 0) {
                    nw /= simulation_days;
                }
            }

            // Calculate goodness of fit
            double ats = AreaTestStat(real_time, simu_time_candidate, NBINS);

            // Update best if this simulation is better
            if (ats < best_ats) {
                best_ats = ats;
                best_index1 = index1;
                best_index2 = index2;
                best_nw = nw;

                // Save this simulation as the best one
                simu_time = simu_time_candidate;
                simu_loc = simu_loc_candidate;
            }
        }
    }

    // Generate location IDs and coordinates for simulation
    simu_locid.clear();
    simu_lon.clear();
    simu_lat.clear();
    simu_day.clear();

    for (size_t i = 0; i < simu_time.size(); i++) {
        // Derive location ID based on location type
        if (simu_loc[i] == 1) { // Home
            simu_locid.push_back(1);
            simu_lon.push_back(home_lon);
            simu_lat.push_back(home_lat);
        } else if (simu_loc[i] == 2) { // Work
            simu_locid.push_back(2);
            simu_lon.push_back(work_lon);
            simu_lat.push_back(work_lat);
        } else { // Other
            simu_locid.push_back(i + 3);
            if (rand() % 2 == 0) {
                simu_lon.push_back(home_lon + (rand() % 100 - 50) * 0.0001);
                simu_lat.push_back(home_lat + (rand() % 100 - 50) * 0.0001);
            } else {
                simu_lon.push_back(work_lon + (rand() % 100 - 50) * 0.0001);
                simu_lat.push_back(work_lat + (rand() % 100 - 50) * 0.0001);
            }
        }

        // Calculate day
        simu_day.push_back(int(timeCorrect(simu_time[i]) / secOneDay));

        // Calculate dt
        if (i > 0) {
            simu_dt.push_back((simu_time[i] - simu_time[i-1]) / 3600.0);
        }
    }

    // Keep track of best trajectories (for global statistics)
    if (real_dt.size() > best_real_dt.size()) {
        best_real_time = real_time;
        best_real_dt = real_dt;
        best_real_loc = real_loc;
        best_real_day = real_day;
        best_real_locid = real_locid;
        best_real_lon = real_lon;
        best_real_lat = real_lat;

        best_simu_time = simu_time;
        best_simu_dt = simu_dt;
        best_simu_loc = simu_loc;
        best_simu_day = simu_day;
        best_simu_locid = simu_locid;
        best_simu_lon = simu_lon;
        best_simu_lat = simu_lat;
    }

    // Write data to output files
    // DNRealCommuters.txt - Real burst sizes (day-night pattern)
    int real_daily_count = 0;
    int real_daily_location_count = 0;
    int real_previous_day = -1;
    bool real_was_home = false;
    bool real_was_work = false;

    for (size_t i = 0; i < real_time.size(); i++) {
        int current_day = int(timeCorrect(real_time[i]) / secOneDay);

        if (real_previous_day != -1) {
            if (current_day == real_previous_day) {
                // Same day
                real_daily_count++;
                if ((real_was_home && real_loc[i] == 1) || (real_was_work && real_loc[i] == 2)) {
                    // Already visited this location type
                } else {
                    real_daily_location_count++;
                }

                if (real_loc[i] == 1) real_was_home = true;
                if (real_loc[i] == 2) real_was_work = true;
            } else {
                // New day - output the previous day
                fprintf(fout_id3, "%d %d\n", real_daily_count, real_daily_location_count);

                // Reset for new day
                real_daily_count = 1;
                real_daily_location_count = 1;
                real_was_home = (real_loc[i] == 1);
                real_was_work = (real_loc[i] == 2);
            }
        } else {
            // First record
            real_daily_count = 1;
            real_daily_location_count = 1;
            real_was_home = (real_loc[i] == 1);
            real_was_work = (real_loc[i] == 2);
        }

        real_previous_day = current_day;
    }

    // Output the last day
    if (real_previous_day != -1) {
        fprintf(fout_id3, "%d %d\n", real_daily_count, real_daily_location_count);
    }

    // DNSimuCommuters.txt - Simulated burst sizes
    int simu_daily_count = 0;
    int simu_daily_location_count = 0;
    int simu_previous_day = -1;
    bool simu_was_home = false;
    bool simu_was_work = false;

    for (size_t i = 0; i < simu_time.size(); i++) {
        int current_day = int(timeCorrect(simu_time[i]) / secOneDay);

        if (simu_previous_day != -1) {
            if (current_day == simu_previous_day) {
                // Same day
                simu_daily_count++;
                if ((simu_was_home && simu_loc[i] == 1) || (simu_was_work && simu_loc[i] == 2)) {
                    // Already visited this location type
                } else {
                    simu_daily_location_count++;
                }

                if (simu_loc[i] == 1) simu_was_home = true;
                if (simu_loc[i] == 2) simu_was_work = true;
            } else {
                // New day - output the previous day
                fprintf(fout_id4, "%d %d\n", simu_daily_count, simu_daily_location_count);

                // Reset for new day
                simu_daily_count = 1;
                simu_daily_location_count = 1;
                simu_was_home = (simu_loc[i] == 1);
                simu_was_work = (simu_loc[i] == 2);
            }
        } else {
            // First record
            simu_daily_count = 1;
            simu_daily_location_count = 1;
            simu_was_home = (simu_loc[i] == 1);
            simu_was_work = (simu_loc[i] == 2);
        }

        simu_previous_day = current_day;
    }

    // Output the last day
    if (simu_previous_day != -1) {
        fprintf(fout_id4, "%d %d\n", simu_daily_count, simu_daily_location_count);
    }

    // SimuLocCommuters.txt - Simulated locations
    for (size_t i = 0; i < simu_time.size(); i++) {
        fprintf(fout_id5, "%s %ld %d %f %d\n",
                regions[0].user_id.c_str(),
                static_cast<long>(simu_time[i]),
                0,  // Always 0 in original code
                double(timeCorrect(simu_time[i])) / secOneDay,
                simu_locid[i]);
    }

    // ParametersCommuters.txt - User parameters
    // Calculate average real dt first
    double avg_real_dt = 0.0;
    int dt_count = 0;

    for (size_t i = 1; i < real_time.size(); i++) {
        avg_real_dt += real_time[i] - real_time[i-1];
        dt_count++;
    }

    if (dt_count > 0) {
        avg_real_dt /= dt_count;
    }

    // Calculate daily location counts
    double param_total_daily_loc_count = 0;
    double param_total_day = 0;
    int param_previous_day = -1;
    int param_daily_loc_count = 0;
    bool param_was_home = false;
    bool param_was_work = false;

    for (size_t i = 0; i < real_time.size(); i++) {
        int current_day = int(timeCorrect(real_time[i]) / secOneDay);

        if (param_previous_day != -1 && current_day != param_previous_day) {
            param_total_daily_loc_count += param_daily_loc_count;
            param_total_day++;
            param_daily_loc_count = 1; // Reset but count current location
            param_was_home = (real_loc[i] == 1);
            param_was_work = (real_loc[i] == 2);
        } else if (param_previous_day == -1) {
            // First record
            param_daily_loc_count = 1;
            param_was_home = (real_loc[i] == 1);
            param_was_work = (real_loc[i] == 2);
        } else {
            // Same day, check if new location type
            if (!(param_was_home && real_loc[i] == 1) &&
                !(param_was_work && real_loc[i] == 2)) {
                param_daily_loc_count++;
            }

            if (real_loc[i] == 1) param_was_home = true;
            if (real_loc[i] == 2) param_was_work = true;
        }

        param_previous_day = current_day;
    }

    // Add last day
    if (param_previous_day != -1) {
        param_total_daily_loc_count += param_daily_loc_count;
        param_total_day++;
    }

    double avg_loc_count = (param_total_day > 0) ?
        (param_total_daily_loc_count / param_total_day * 1.3) : 0;

    // Make sure the fprintf statement uses the correct variables
    fprintf(fout_id6, "%d %d %f %f %f %f %f %f %s\n",
            best_index1, best_index2, best_nw, avg_loc_count,
            home_lon, home_lat, work_lon, work_lat,
            regions[0].user_id.c_str());

    // RealLocCommuters.txt - Real locations
    for (size_t i = 0; i < real_time.size(); i++) {
        fprintf(fout_id7, "%s %ld %d %f %d\n",
                regions[0].user_id.c_str(),
                static_cast<long>(real_time[i]),
                real_loc[i],
                double(timeCorrect(real_time[i])) / secOneDay,
                real_locid[i]);
    }

    // Update for DTReal and DTSimu files (these will be written at the end)
    RescaleTime(real_time, true);
    RescaleTime(simu_time, false);

    // Make sure everything gets written
    fflush(fout_id3);
    fflush(fout_id4);
    fflush(fout_id5);
    fflush(fout_id6);
    fflush(fout_id7);
}

// Process a batch of users
void process_users_batch(std::unordered_map<std::string, std::vector<StayRegion>>& user_data,
                       FILE* fout_id1, FILE* fout_id2, FILE* fout_id3,
                       FILE* fout_id4, FILE* fout_id5, FILE* fout_id6, FILE* fout_id7,
                       int slot_interval) {
    for (auto& [user_id, regions] : user_data) {
        // Sort regions by timestamp
        std::sort(regions.begin(), regions.end(),
                [](const StayRegion& a, const StayRegion& b) {
                    return a.timestamp < b.timestamp;
                });
        process_single_user(regions, fout_id1, fout_id2, fout_id3, fout_id4, fout_id5, fout_id6, fout_id7, slot_interval);
    }

    // Clear processed data to free memory
    user_data.clear();
}

// Function to process data in batches
void process_stay_regions_streaming(const std::string& input_path,
                                   FILE* fout_id1, FILE* fout_id2, FILE* fout_id3,
                                   FILE* fout_id4, FILE* fout_id5, FILE* fout_id6, FILE* fout_id7,
                                   int slot_interval) {

    // Add at the start of process_stay_regions_streaming function
    std::cout << "File path length: " << input_path.length() << std::endl;
    std::cout << "File path: " << input_path << std::endl;

    // STANDALONE RUNS WITHOUT THIS
    // static bool arrow_initialized = false;
    // if (!arrow_initialized) {
    //     // Initialize Arrow's default memory pool
    //     auto status = arrow::MemoryPool::InitializeDefaultPool();
    //     if (!status.ok()) {
    //         std::cerr << "Failed to initialize Arrow memory pool: " << status.ToString() << std::endl;
    //     }
    //     arrow_initialized = true;
    //     std::cout << "Arrow initialized successfully" << std::endl;
    // }

    // Open the file
    auto memory_pool = arrow::MemoryPool::CreateDefault();
    auto maybe_infile = arrow::io::ReadableFile::Open(input_path, memory_pool.get());
    if (!maybe_infile.ok()) {
        std::cerr << "Could not open file: " << input_path << std::endl;
        return;
    }
    std::shared_ptr<arrow::io::ReadableFile> infile = maybe_infile.ValueOrDie();

    // Create a ParquetFileReader
    auto maybe_reader = parquet::arrow::OpenFile(infile, memory_pool.get());
    if (!maybe_reader.ok()) {
        std::cerr << "Could not open Parquet file: " << input_path << " - Error: " << maybe_reader.status().ToString() << std::endl;
        return;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(maybe_reader).ValueOrDie();

    // Get file metadata
    auto file_metadata = reader->parquet_reader()->metadata();
    int num_row_groups = file_metadata->num_row_groups();
    std::cout << "Number of row groups: " << num_row_groups << std::endl;

    // Track unique users across all batches
    std::unordered_map<std::string, std::vector<StayRegion>> user_data;
    int total_users_processed = 0;
    int total_records_processed = 0;

    // Process each row group
    for (int group = 0; group < num_row_groups; group++) {
        try {
            if (!g_quiet) {
                std::cout << "Processing row group " << group + 1 << " of " << num_row_groups << std::endl;
            }

            // Read entire row group into a Table
            std::shared_ptr<arrow::Table> table;
            auto status = reader->ReadRowGroup(group, &table);
            if (!status.ok()) {
                std::cerr << "Error reading row group " << group << ": " << status.ToString() << std::endl;
                continue;
            }

            // Get row count
            int64_t num_rows = table->num_rows();
            if (!g_quiet) {
                std::cout << "Row group has " << num_rows << " rows" << std::endl;

                // Print schema for debugging
                std::cout << "Table schema: " << std::endl;
                for (int i = 0; i < table->schema()->num_fields(); i++) {
                    const auto& field = table->schema()->field(i);
                    std::cout << "  " << i << ": " << field->name() << " (Type: " << field->type()->ToString() << ")" << std::endl;
                }
            }

            // Find relevant columns by name
            int user_id_idx = -1, timestamp_idx = -1, type_idx = -1, day_idx = -1;
            int work_h3_idx = -1, home_h3_idx = -1, lat_idx = -1, lon_idx = -1, locid_idx = -1;

            for (int i = 0; i < table->schema()->num_fields(); i++) {
                std::string field_name = table->schema()->field(i)->name();
                std::transform(field_name.begin(), field_name.end(), field_name.begin(), ::tolower);

                if (field_name == "caid" || field_name == "user_id" || field_name == "userid")
                    user_id_idx = i;
                else if (field_name == "stay_start_timestamp" || field_name == "timestamp")
                    timestamp_idx = i;
                else if (field_name == "type")
                    type_idx = i;
                else if (field_name == "day_of_week" || field_name == "day")
                    day_idx = i;
                else if (field_name == "work_h3_index" || field_name == "work_h3")
                    work_h3_idx = i;
                else if (field_name == "home_h3_index" || field_name == "home_h3")
                    home_h3_idx = i;
                else if (field_name == "latitude" || field_name == "lat")
                    lat_idx = i;
                else if (field_name == "longitude" || field_name == "lon")
                    lon_idx = i;
                else if (field_name == "h3_region_stay_id" || field_name == "location_id")
                    locid_idx = i;
            }

            std::cout << "Found columns: "
                      << "user_id=" << user_id_idx << ", "
                      << "timestamp=" << timestamp_idx << ", "
                      << "type=" << type_idx << ", "
                      << "day=" << day_idx << ", "
                      << "locid=" << locid_idx << ", "
                      << "work_h3=" << work_h3_idx << ", "
                      << "home_h3=" << home_h3_idx << std::endl;

            // Check if required columns are found
            if (user_id_idx == -1 || timestamp_idx == -1) {
                std::cerr << "Required columns not found. Skipping file." << std::endl;
                continue;
            }

            // Process each chunk in the table
            int num_chunks = table->column(user_id_idx)->num_chunks();
            std::cout << "Number of chunks: " << num_chunks << std::endl;

            for (int c = 0; c < num_chunks; c++) {
                // Get arrays for this chunk
                std::shared_ptr<arrow::Array> user_id_array = table->column(user_id_idx)->chunk(c);
                std::shared_ptr<arrow::Array> timestamp_array = table->column(timestamp_idx)->chunk(c);

                // Get array types
                arrow::Type::type user_id_type = user_id_array->type_id();
                arrow::Type::type timestamp_type = timestamp_array->type_id();

                std::cout << "Chunk " << c << " types - "
                          << "user_id: " << user_id_type << ", "
                          << "timestamp: " << timestamp_type << std::endl;

                // Handle user_id (support multiple types)
                std::shared_ptr<arrow::StringArray> string_user_ids;
                std::shared_ptr<arrow::Int32Array> int32_user_ids;
                std::shared_ptr<arrow::Int64Array> int64_user_ids;

                bool user_id_is_string = false;
                bool user_id_is_int32 = false;
                bool user_id_is_int64 = false;

                if (user_id_type == arrow::Type::STRING) {
                    user_id_is_string = true;
                    string_user_ids = std::dynamic_pointer_cast<arrow::StringArray>(user_id_array);
                    if (!string_user_ids) {
                        std::cerr << "Failed to cast user_id to StringArray" << std::endl;
                        continue;
                    }
                } else if (user_id_type == arrow::Type::INT32) {
                    user_id_is_int32 = true;
                    int32_user_ids = std::dynamic_pointer_cast<arrow::Int32Array>(user_id_array);
                    if (!int32_user_ids) {
                        std::cerr << "Failed to cast user_id to Int32Array" << std::endl;
                        continue;
                    }
                } else if (user_id_type == arrow::Type::INT64) {
                    user_id_is_int64 = true;
                    int64_user_ids = std::dynamic_pointer_cast<arrow::Int64Array>(user_id_array);
                    if (!int64_user_ids) {
                        std::cerr << "Failed to cast user_id to Int64Array" << std::endl;
                        continue;
                    }
                }

                // Handle timestamp (support TIMESTAMP type)
                std::shared_ptr<arrow::TimestampArray> timestamp_ts_array;
                std::shared_ptr<arrow::Int64Array> int64_timestamps;
                std::shared_ptr<arrow::Int32Array> int32_timestamps;

                bool timestamp_is_timestamp = false;
                bool timestamp_is_int64 = false;
                bool timestamp_is_int32 = false;

                if (timestamp_type == arrow::Type::TIMESTAMP) {
                    timestamp_is_timestamp = true;
                    timestamp_ts_array = std::dynamic_pointer_cast<arrow::TimestampArray>(timestamp_array);
                    if (!timestamp_ts_array) {
                        std::cerr << "Failed to cast timestamp to TimestampArray" << std::endl;
                        continue;
                    }
                    std::cout << "Successfully cast to TimestampArray" << std::endl;
                } else if (timestamp_type == arrow::Type::INT64) {
                    timestamp_is_int64 = true;
                    int64_timestamps = std::dynamic_pointer_cast<arrow::Int64Array>(timestamp_array);
                    if (!int64_timestamps) {
                        std::cerr << "Failed to cast timestamp to Int64Array" << std::endl;
                        continue;
                    }
                } else if (timestamp_type == arrow::Type::INT32) {
                    timestamp_is_int32 = true;
                    int32_timestamps = std::dynamic_pointer_cast<arrow::Int32Array>(timestamp_array);
                    if (!int32_timestamps) {
                        std::cerr << "Failed to cast timestamp to Int32Array" << std::endl;
                        continue;
                    }
                } else {
                    std::cerr << "Unsupported timestamp type: " << timestamp_type << std::endl;
                    continue;
                }

                // Optional columns
                std::shared_ptr<arrow::Array> type_array =
                    (type_idx >= 0) ? table->column(type_idx)->chunk(c) : nullptr;
                std::shared_ptr<arrow::Array> day_array =
                    (day_idx >= 0) ? table->column(day_idx)->chunk(c) : nullptr;
                std::shared_ptr<arrow::Array> work_h3_array =
                    (work_h3_idx >= 0) ? table->column(work_h3_idx)->chunk(c) : nullptr;
                std::shared_ptr<arrow::Array> home_h3_array =
                    (home_h3_idx >= 0) ? table->column(home_h3_idx)->chunk(c) : nullptr;
                std::shared_ptr<arrow::Array> lat_array =
                    (lat_idx >= 0) ? table->column(lat_idx)->chunk(c) : nullptr;
                std::shared_ptr<arrow::Array> lon_array =
                    (lon_idx >= 0) ? table->column(lon_idx)->chunk(c) : nullptr;

                // Process rows in this chunk
                int64_t chunk_size = user_id_array->length();
                std::cout << "Processing " << chunk_size << " rows in chunk " << c << std::endl;

                for (int64_t i = 0; i < chunk_size; i++) {
                    // Skip null values
                    if (user_id_array->IsNull(i) || timestamp_array->IsNull(i)) {
                        continue;
                    }

                    StayRegion region;

                    // Get user_id (handle different types)
                    if (user_id_is_string) {
                        region.user_id = string_user_ids->GetString(i);
                    } else if (user_id_is_int32) {
                        region.user_id = std::to_string(int32_user_ids->Value(i));
                    } else if (user_id_is_int64) {
                        region.user_id = std::to_string(int64_user_ids->Value(i));
                    }

                    // Get timestamp (handle different types)
                    if (timestamp_is_timestamp) {
                        // Convert nanoseconds to seconds
                        region.timestamp = timestamp_ts_array->Value(i) / 1000000000;
                    } else if (timestamp_is_int64) {
                        region.timestamp = int64_timestamps->Value(i);
                    } else if (timestamp_is_int32) {
                        region.timestamp = static_cast<int64_t>(int32_timestamps->Value(i));
                    }

                    // Add location type if available
                    if (type_array && !type_array->IsNull(i)) {
                        if (auto int_array = std::dynamic_pointer_cast<arrow::Int32Array>(type_array)) {
                            region.location_type = int_array->Value(i);
                        }
                    }

                    // Add day of week if available
                    if (day_array && !day_array->IsNull(i)) {
                        if (auto int_array = std::dynamic_pointer_cast<arrow::Int32Array>(day_array)) {
                            region.day_of_week = int_array->Value(i);
                        }
                    }

                    // Add work_h3 if available
                    if (work_h3_array && !work_h3_array->IsNull(i)) {
                        if (auto string_array = std::dynamic_pointer_cast<arrow::StringArray>(work_h3_array)) {
                            region.work_h3_index = string_array->GetString(i);
                        }
                    }

                    // Add home_h3 if available
                    if (home_h3_array && !home_h3_array->IsNull(i)) {
                        if (auto string_array = std::dynamic_pointer_cast<arrow::StringArray>(home_h3_array)) {
                            region.home_h3_index = string_array->GetString(i);
                        }
                    }

                    // Add lat/lon if available
                    if (lat_array && !lat_array->IsNull(i)) {
                        if (auto double_array = std::dynamic_pointer_cast<arrow::DoubleArray>(lat_array)) {
                            region.lat = double_array->Value(i);
                        }
                    }

                    if (lon_array && !lon_array->IsNull(i)) {
                        if (auto double_array = std::dynamic_pointer_cast<arrow::DoubleArray>(lon_array)) {
                            region.lon = double_array->Value(i);
                        }
                    }

                    // Add location_id if available
                    int location_id = i; // Default to index
                    if (locid_idx != -1) {
                        std::shared_ptr<arrow::Array> locid_array = table->column(locid_idx)->chunk(c);
                        if (locid_array && !locid_array->IsNull(i)) {
                            if (auto int_array = std::dynamic_pointer_cast<arrow::Int32Array>(locid_array)) {
                                location_id = int_array->Value(i);
                            } else if (auto int64_array = std::dynamic_pointer_cast<arrow::Int64Array>(locid_array)) {
                                location_id = static_cast<int>(int64_array->Value(i));
                            }
                        }
                    }
                    region.location_id = location_id;

                    // Add to user data
                    user_data[region.user_id].push_back(region);
                    total_records_processed++;

                    // Process in batches to manage memory
                    if (user_data.size() >= CHUNK_SIZE) {
                        total_users_processed += user_data.size();
                        std::cout << "Processing batch of " << user_data.size() << " users..." << std::endl;
                        process_users_batch(user_data, fout_id1, fout_id2, fout_id3, fout_id4, fout_id5, fout_id6, fout_id7, slot_interval);
                        user_data.clear();
                    }
                }
            }

            // Process remaining users
            if (!user_data.empty()) {
                std::cout << "Processing final batch of " << user_data.size() << " users..." << std::endl;
                total_users_processed += user_data.size();
                process_users_batch(user_data, fout_id1, fout_id2, fout_id3, fout_id4, fout_id5, fout_id6, fout_id7, slot_interval);
                user_data.clear();
            }

        } catch (const std::exception& e) {
            std::cerr << "Error processing row group " << group << ": " << e.what() << std::endl;
        }
    }

    std::cout << "\nProcessing complete:" << std::endl;
    std::cout << "Total users processed: " << total_users_processed << std::endl;
    std::cout << "Total records processed: " << total_records_processed << std::endl;
}

// Main processing function
void process_streaming(
    const std::string& input_path,
    const std::string& output_dir,
    bool commuter_mode,
    int min_num_stay,
    int max_num_stay,
    double nw_thres,
    int slot_interval,
    double rho,
    double gamma
) {
    // Define output file names based on commuter mode
    std::string file_name1, file_name2, file_name3, file_name4, file_name5, file_name6, file_name7;

    std::string activity_file1, activity_file2, activity_file3;

    if (commuter_mode) {
        activity_file1 = "Comm_pt_daily.txt";
        activity_file2 = "Comm_pt_weekly.txt";
        activity_file3 = "Comm_pt_daily_weekly.txt";
    } else {
        activity_file1 = "NonComm_pt_daily.txt";
        activity_file2 = "NonComm_pt_weekly.txt";
        activity_file3 = "NonComm_pt_daily_weekly.txt";
    }

    if (commuter_mode) {
        file_name1 = output_dir + "/Commuters/DTRealCommuters.txt";
        file_name2 = output_dir + "/Commuters/DTSimuCommuters.txt";
        file_name3 = output_dir + "/Commuters/DNRealCommuters.txt";
        file_name4 = output_dir + "/Commuters/DNSimuCommuters.txt";
        file_name5 = output_dir + "/Commuters/SimuLocCommuters.txt";
        file_name6 = output_dir + "/Commuters/ParametersCommuters.txt";
        file_name7 = output_dir + "/Commuters/RealLocCommuters.txt";
    } else {
        file_name1 = output_dir + "/NonCommuters/DTRealNonCommuters.txt";
        file_name2 = output_dir + "/NonCommuters/DTSimuNonCommuters.txt";
        file_name3 = output_dir + "/NonCommuters/DNRealNonCommuters.txt";
        file_name4 = output_dir + "/NonCommuters/DNSimuNonCommuters.txt";
        file_name5 = output_dir + "/NonCommuters/SimuLocNonCommuters.txt";
        file_name6 = output_dir + "/NonCommuters/ParametersNonCommuters.txt";
        file_name7 = output_dir + "/NonCommuters/RealLocNonCommuters.txt";
    }

    // Open output files
    FILE* fout_id1 = fopen(file_name1.c_str(), "w");
    FILE* fout_id2 = fopen(file_name2.c_str(), "w");
    FILE* fout_id3 = fopen(file_name3.c_str(), "w");
    FILE* fout_id4 = fopen(file_name4.c_str(), "w");
    FILE* fout_id5 = fopen(file_name5.c_str(), "w");
    FILE* fout_id6 = fopen(file_name6.c_str(), "w");
    FILE* fout_id7 = fopen(file_name7.c_str(), "w");


    // Open activity pattern files
    FILE* activity_out1 = fopen(activity_file1.c_str(), "w");
    FILE* activity_out2 = fopen(activity_file2.c_str(), "w");
    FILE* activity_out3 = fopen(activity_file3.c_str(), "w");

    if (!activity_out1 || !activity_out2 || !activity_out3) {
        std::cerr << "Error opening activity pattern files" << std::endl;
        return;
    }

    // Get all Parquet files (filter out .crc files)
    std::vector<std::string> parquet_files = get_parquet_files(input_path);

    // Process each file using the streaming approach
    std::cout << "About to process " << parquet_files.size() << " files..." << std::endl;

    if (commuter_mode) {
        fs::create_directories(output_dir + "/Commuters");
    } else {
        fs::create_directories(output_dir + "/NonCommuters");
    }

    for (const auto& file : parquet_files) {
        std::cout << "\nProcessing file: " << file << std::endl;
        process_stay_regions_streaming(file, fout_id1, fout_id2, fout_id3, fout_id4, fout_id5, fout_id6, fout_id7, slot_interval);
    }

    std::cout << "Finished processing all " << parquet_files.size() << " files." << std::endl;

    // Output real and simulated dt data
    std::cout << "Writing aggregated data to output files..." << std::endl;

    // Write accumulated time data to DTRealCommuters.txt and DTSimuCommuters.txt
    for (size_t i = 0; i < original_real_dt.size(); i++) {
        fprintf(fout_id1, "%f %f\n", original_real_dt[i], rescaled_real_dt[i]);
    }

    for (size_t i = 0; i < original_simu_dt.size(); i++) {
        fprintf(fout_id2, "%f %f\n", original_simu_dt[i], rescaled_simu_dt[i]);
    }

    // Normalize and write activity patterns
    double total_activities = 0.0;
    for (const auto& val : daily_activeness) {
        total_activities += val;
    }

    if (total_activities > 0) {
        // Write and normalize daily patterns
        for (auto& val : daily_activeness) {
            val /= total_activities;
            fprintf(activity_out1, "%f\n", val);
        }

        // Write and normalize weekly patterns
        for (auto& val : weekly_activeness) {
            val /= total_activities;
            fprintf(activity_out2, "%f\n", val);
        }

        // Write and normalize daily-weekly patterns
        for (auto& val : daily_weekly_activeness) {
            val /= total_activities;
            fprintf(activity_out3, "%f\n", val);
        }
    }

    // Close activity pattern files
    fclose(activity_out1);
    fclose(activity_out2);
    fclose(activity_out3);

    // Close all output files
    fclose(fout_id1);
    fclose(fout_id2);
    fclose(fout_id3);
    fclose(fout_id4);
    fclose(fout_id5);
    fclose(fout_id6);
    fclose(fout_id7);
}

double AreaTestStat(const std::vector<double>& times, const std::vector<double>& simulated_time, int nbins) {
    if (times.empty() || simulated_time.empty()) {
        return std::numeric_limits<double>::max();
    }

    // Find min and max values across both distributions
    double min_time = std::numeric_limits<double>::max();
    double max_time = std::numeric_limits<double>::lowest();

    for (const auto& t : times) {
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }

    for (const auto& t : simulated_time) {
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }

    double range = max_time - min_time;
    if (range <= 0) {
        return std::numeric_limits<double>::max();
    }

    // Create the bins for binned CDF
    std::vector<double> real_cdf(nbins, 0.0);
    std::vector<double> simu_cdf(nbins, 0.0);

    // Count elements in each bin for real data
    for (const auto& t : times) {
        int bin = std::min(static_cast<int>((t - min_time) / range * nbins), nbins - 1);
        real_cdf[bin]++;
    }

    // Count elements in each bin for simulated data
    for (const auto& t : simulated_time) {
        int bin = std::min(static_cast<int>((t - min_time) / range * nbins), nbins - 1);
        simu_cdf[bin]++;
    }

    // Convert counts to CDF
    double real_total = times.size();
    double simu_total = simulated_time.size();

    double real_cum = 0;
    double simu_cum = 0;

    for (int i = 0; i < nbins; i++) {
        real_cum += real_cdf[i] / real_total;
        simu_cum += simu_cdf[i] / simu_total;

        real_cdf[i] = real_cum;
        simu_cdf[i] = simu_cum;
    }

    // Calculate area between the CDFs
    double area = 0.0;
    for (int i = 0; i < nbins; i++) {
        area += std::abs(real_cdf[i] - simu_cdf[i]);
    }

    // Normalize by number of bins to get average difference
    return area / nbins;
}

void run_DT_simulation(
    const std::string& input_path,
    const std::string& output_dir,
    bool commuter_mode,
    int min_num_stay,
    int max_num_stay,
    double nw_thres,
    int slot_interval,
    double rho,
    double gamma
) {
    // Set memory limit to 2GB
    struct rlimit rl;
    rl.rlim_cur = 2048L * 1024 * 1024;
    rl.rlim_max = 2048L * 1024 * 1024;
    setrlimit(RLIMIT_AS, &rl);

    srand(time(NULL));

    // Create output directories
    if (commuter_mode) {
        fs::create_directories(output_dir + "/Commuters");
    } else {
        fs::create_directories(output_dir + "/NonCommuters");
    }

    // Call process_streaming which handles all the file processing
    process_streaming(
        input_path,
        output_dir,
        commuter_mode,
        min_num_stay,
        max_num_stay,
        nw_thres,
        slot_interval,
        rho,
        gamma
    );

    std::cout << "Processing completed successfully." << std::endl;
}

// CLI main
int main(int argc, char* argv[]) {
    // --quiet may appear anywhere after the required positional args; also
    // honor TIMEGEO_QUIET=1 so callers who can't edit argv can silence us.
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quiet") { g_quiet = true; continue; }
        if (a == "--verbose") { g_quiet = false; continue; }
        pos.push_back(a);
    }
    if (const char* env = std::getenv("TIMEGEO_QUIET")) {
        if (std::string(env) == "1") g_quiet = true;
    }

    if (!g_quiet) std::cout << "Starting program..." << std::endl;

    std::string input_path = pos[0];
    std::string output_dir = pos[1];
    bool commuter_flag = pos.size() > 2 ? (std::stoi(pos[2]) != 0) : true;
    int min_num_stay  = pos.size() > 3 ? std::stoi(pos[3]) : 2;
    int max_num_stay  = pos.size() > 4 ? std::stoi(pos[4]) : 3000;
    double nw_thres   = pos.size() > 5 ? std::stod(pos[5]) : 1.0;
    int slot_interval = pos.size() > 6 ? std::stoi(pos[6]) : 600;
    double rho        = pos.size() > 7 ? std::stod(pos[7]) : 0.6;
    double gamma      = pos.size() > 8 ? std::stod(pos[8]) : -0.21;
    run_DT_simulation(input_path, output_dir, commuter_flag, min_num_stay, max_num_stay, nw_thres, slot_interval, rho, gamma);
    if (!g_quiet) std::cout << "Ending program..." << std::endl;
}
