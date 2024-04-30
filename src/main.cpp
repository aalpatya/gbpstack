/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#define RLIGHTS_IMPLEMENTATION // needed to be defined once for the lights shader

#include <DArgs/DArgs.h>
#include <Eigen/Dense>
#include <raylib.h>
#include "json.hpp"

#include <gbp/utils.h>
#include <gbp/globals.h>
#include <gbp/Robot.h>
#include <gbp/Simulator.h>
Globals globals;

void log_results(Simulator* sim);

int main(int argc, char *argv[]){
    // Clock for measuring performance
    auto start = std::chrono::steady_clock::now();

    // Set random seed 
    srand((int)globals.SEED);

    // Parse arguments from config.json file
    DArgs::DArgs dargs(argc, argv);
    if (globals.parse_global_args(dargs)) return EXIT_FAILURE;

    // Create simulator with initial formation from config.json.
    Simulator* sim = new Simulator(globals.INITIAL_FORMATION);
    while (globals.RUN){
        // Handle input, and create / delete robots if necessary
        sim->updateSetup();

        // Timestep
        sim->timestep();

        // Check simulation ending conditions
        if (sim->clock_>=globals.MAX_TIME) {globals.RESET = false; globals.RUN = false;}
        
        // Ending conditions for source seeking or coverage experiments
        if (sim->source_found_ || (sim->full_coverage_ && ((sim->clock_- sim->time_finished_)>globals.STOPPING_TIME_AFTER_FINISHED))) {
            globals.RESET = false; globals.RUN = false;
        };

        // Display
        if (globals.CAPTURE) sim->draw();

    }

    // // Log results to file
    log_results(sim);

    std::cout << "Elapsed(us): " << since(start).count()/(float)1e6 << std::endl;    
    delete sim;

    return 0;
}    

void log_results(Simulator* sim){
    // Recording/logging output code
    print("Background name: "+globals.BG_NAME, "Seed: "+globals.SEED, "Time taken for all robots to finish: "+sim->clock_);
    globals.TESTNAME = (globals.TESTNAME.empty()) ? globals.BG_NAME + "_" + std::to_string(globals.SEED) : globals.TESTNAME;
    globals.jsonfile_out[globals.TESTNAME]["time"] = sim->clock_;
    globals.jsonfile_out[globals.TESTNAME]["rms"] = sim->rms_vector_;
    globals.jsonfile_out[globals.TESTNAME]["coverage"] = sim->coverage_vector_; 
    globals.jsonfile_out[globals.TESTNAME]["bg_file"] = globals.BACKGROUND_FILE; 
    globals.jsonfile_out[globals.TESTNAME]["time_finished"] = sim->time_finished_; 
    std::vector<std::vector<Eigen::Vector2d>>positions{};
    std::vector<float> distances_travelled{};
    for (auto [rid, robot] : sim->robots_){
        globals.jsonfile_out[globals.TESTNAME]["robots"][rid] = robot->positions_; 
        globals.jsonfile_out[globals.TESTNAME]["distances"][rid] = robot->distance_travelled_; 
    }           
    std::ofstream file(globals.OUTPUT_FILE); file << globals.jsonfile_out;  
    print("Results in: ", globals.OUTPUT_FILE, globals.jsonfile_out[globals.TESTNAME]["time"]);

}