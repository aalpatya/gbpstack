/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include <cmath>
#include <raylib.h>
#include <DArgs/DArgs.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include <raymath.h>
#include <rcamera.h>
#include "json.hpp"
/************************ Enums and structs ************************/
enum NodeType {VARIABLE_NODE, FACTOR_NODE};
enum FactorType {NONE_FACTOR, DYNAMICS_FACTOR, INTERROBOT_FACTOR, OBSTACLE_FACTOR, ATTRACTOR_FACTOR, CLOSENESS_FACTOR, RIGIDLINK_FACTOR, RANGEBEARING_FACTOR, UNARY_FACTOR};
enum Scenario {ConstantVelocity, Collaborative};

enum MODES_LIST {SimNone, Timestep, OneTimestep, Iterate, OneIteration, Help, Junction};
enum MOUSE_MODES_LIST {MouseNone, Obstacle, Attractor, Swarm, Draw, AddRobotStart, AddRobotGoal, Eraser, RobotStartStop};

enum FormationName{
    FORMATION_SWARM_CORNER,
    FORMATION_SWARM_RANDOM,
    FORMATION_SWARM_LINE,
    FORMATION_SWARM_CIRCLE,
    FORMATION_LAST
};

enum class FGLayer {Planning, Information, Goal};

enum class StateTag {Current, Intermediate, Horizon};

class Globals {
    public:
    bool RESET = true;
    bool RUN = true;
    nlohmann::json jsonfile_out;

    std::string CONFIG_FILE = "../config/config.json";
    std::string BACKGROUND_FILE;
    std::string BG_NAME;
    std::string OUTPUT_FILE;
    bool TAKE_SCREENSHOTS = false;
    std::string OUTPUT_SCREENSHOTS;
    std::string TESTNAME;
    const char* WINDOW_TITLE = "Distributing Multirobot Motion Planning with Gaussian Belief Propogation";
    int CAPTURE;
    int SEED;
    int N_DOFS = 4;     // dofs
    int WORLD_SZ;   // [pixels]
    int SCREEN_SZ;  // [pixels]
    bool DRAW_FAC;
    bool DRAW_PATH;
    bool DRAW_INFO;
    bool DRAW_GOAL;
    float DROPOUT;
    int MODE;


    int NUM_ROBOTS;
    int ROBOT_MAX_NEIGHBOURS;
    int NUM_OBSTACLES;
    float T_HORIZON;
    int SAMPLE_TIMESTEP;
    int CELL_WIDTH;
    float ROBOT_RADIUS;                    // [m]
    float COMMUNICATION_RADIUS;                   // [m]
    float MAPPING_RADIUS;                   // [m]
    int MAX_TIME;
    int STOPPING_TIME_AFTER_FINISHED;
    float MAX_SPEED;      // [m/s]
    float TIMESTEP;
    int NUM_ITERS;

    float SIGMA_POSE_FIXED;             // For the current state
    float SIGMA_POSE_END;                // For the horizon state
    float SIGMA_FACTOR_DYNAMICS;           // For dynamics factors
    float SIGMA_FACTOR_INTERROBOT;       // Interrobot factor strength
    float SIGMA_FACTOR_OBSTACLE;         // Strength for the static obstacles
    float SIGMA_FACTOR_MAP;         // Strength for the attractor field   
    float SIGMA_FACTOR_INTERROBOT_GOAL;
    float SIGMA_FACTOR_SEEKING_GOAL;
    float SIGMA_FACTOR_MOTION_GOAL;
    float SIGMA_FACTOR_COVERAGE_GOAL;
    float SIGMA_FACTOR_CONSENSUS;

    float p1;
    float p2;
    float p3;
    float p4;
    float GRAVITY = -10.f;
    
    bool PLANNING = true;
    MODES_LIST SIM_MODE = Timestep, LAST_SIM_MODE;
    MOUSE_MODES_LIST MOUSE_MODE;    


    float EPS;
    float MIN_INTERNODAL_DIST;
    float D0;
    float T0    ;

    // Initial formation
    FormationName INITIAL_FORMATION = FORMATION_SWARM_CORNER;
    
    Globals();

    int parse_global_args(DArgs::DArgs& dargs);
    void parse_global_args(std::ifstream& config_file);
    void post_parsing();
    void update_camera();

    float DAMPING = 0.;
    float ROTATION_ANGLE = 0.;
    int LOOKAHEAD_MULTIPLE = 2;
    float k=0; // attractor scaling
    Eigen::Vector2d danger {{0., 0.}};
    bool camera_flag = false;

    int gn; // num square len for informatino display

};