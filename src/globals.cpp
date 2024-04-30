/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <gbp/globals.h>
#include <gbp/utils.h>
#include "json.hpp"
#include <tuple>

#define getName(VariableName) # VariableName
Globals::Globals(){

};

void Globals::parse_global_args(std::ifstream& config_file){
    nlohmann::json j;
    config_file >> j;

    TESTNAME = j["TESTNAME"];
    SEED = j["SEED"];
    CAPTURE = j["CAPTURE"];
    N_DOFS = j["N_DOFS"];
    WORLD_SZ = j["WORLD_SZ"];
    SCREEN_SZ = j["SCREEN_SZ"];
    DRAW_FAC = static_cast<bool>((int)j["DRAW_FAC"]);
    DRAW_PATH = static_cast<bool>((int)j["DRAW_PATH"]);
    DRAW_INFO = static_cast<bool>((int)j["DRAW_INFO"]);
    DRAW_GOAL = static_cast<bool>((int)j["DRAW_GOAL"]);

    INITIAL_FORMATION = static_cast<FormationName>(j["INITIAL_FORMATION"]);
    ROBOT_MAX_NEIGHBOURS = j["ROBOT_MAX_NEIGHBOURS"];
    NUM_ROBOTS = j["NUM_ROBOTS"];
    NUM_OBSTACLES = j["NUM_OBSTACLES"];
    T_HORIZON = j["T_HORIZON"];
    SAMPLE_TIMESTEP = j["SAMPLE_TIMESTEP"];
    CELL_WIDTH = j["CELL_WIDTH"];
    ROBOT_RADIUS = j["ROBOT_RADIUS"];
    COMMUNICATION_RADIUS = j["COMMUNICATION_RADIUS"];
    MAPPING_RADIUS = j["MAPPING_RADIUS"];
    MAX_TIME = j["MAX_TIME"];
    STOPPING_TIME_AFTER_FINISHED = j["STOPPING_TIME_AFTER_FINISHED"];
    NUM_ITERS = j["NUM_ITERS"];
    MAX_SPEED = j["MAX_SPEED"];
    TIMESTEP = j["TIMESTEP"];
    SIGMA_POSE_FIXED = j["SIGMA_POSE_FIXED"];
    SIGMA_POSE_END = j["SIGMA_POSE_END"];
    SIGMA_FACTOR_DYNAMICS = j["SIGMA_FACTOR_DYNAMICS"];
    SIGMA_FACTOR_INTERROBOT = j["SIGMA_FACTOR_INTERROBOT"];
    SIGMA_FACTOR_OBSTACLE = j["SIGMA_FACTOR_OBSTACLE"];
    SIGMA_FACTOR_MAP = j["SIGMA_FACTOR_MAP"];
    SIGMA_FACTOR_INTERROBOT_GOAL = j["SIGMA_FACTOR_INTERROBOT_GOAL"];
    SIGMA_FACTOR_SEEKING_GOAL = j["SIGMA_FACTOR_SEEKING_GOAL"];
    SIGMA_FACTOR_MOTION_GOAL = j["SIGMA_FACTOR_MOTION_GOAL"];
    SIGMA_FACTOR_COVERAGE_GOAL = j["SIGMA_FACTOR_COVERAGE_GOAL"];
    SIGMA_FACTOR_CONSENSUS = j["SIGMA_FACTOR_CONSENSUS"];
    BACKGROUND_FILE = PROJECT_DIR+std::string(j["BACKGROUND_FILE"]);
    OUTPUT_FILE = PROJECT_DIR+std::string(j["OUTPUT_FILE"]);
    TAKE_SCREENSHOTS = static_cast<bool>((int)j["TAKE_SCREENSHOTS"]);
    OUTPUT_SCREENSHOTS = PROJECT_DIR+std::string(j["OUTPUT_SCREENSHOTS"]);
    BG_NAME = j["BG_NAME"];
    DROPOUT = j["DROPOUT"];
    MODE = j["MODE"];

    p1 = j["p1"];
    p2 = j["p2"];
    p3 = j["p3"];
    p4 = j["p4"];
    gn = WORLD_SZ/CELL_WIDTH;

}

int Globals::parse_global_args(DArgs::DArgs &dargs)
{
    // Argument parser
    this->CONFIG_FILE = dargs("--cfg", "config_file", this->CONFIG_FILE);
    
    if (!dargs.check())
    {
        dargs.print_help();
        print("Incorrect arguments!");
        return EXIT_FAILURE;
    }

    std::ifstream my_config_file(CONFIG_FILE);
    assert(my_config_file && "Couldn't find the config file");
    parse_global_args(my_config_file);
    post_parsing();

    return 0;
};

void Globals::post_parsing()
{
    MIN_INTERNODAL_DIST = ROBOT_RADIUS/2.f;
    // Cap max speed, since it should be <= MIN_INTERNODAL_DIST / TIMESTEP
    if (MAX_SPEED > MIN_INTERNODAL_DIST/TIMESTEP){
        MAX_SPEED = MIN_INTERNODAL_DIST/TIMESTEP;
        print("Capping MAX_SPEED parameter at ", MAX_SPEED);
    }
    T0 = MIN_INTERNODAL_DIST / MAX_SPEED; // Time duration between current state and first future state

    // READ contents from pre-existing output file.
    std::ifstream config_file_out(OUTPUT_FILE);
    assert(config_file_out && "Couldn't find the config file");
    if (config_file_out.peek() != std::ifstream::traits_type::eof()){
        config_file_out >> jsonfile_out;
    }
}
