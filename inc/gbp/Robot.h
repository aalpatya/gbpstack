/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include "Simulator.h"
#include <raymath.h>
#include <memory>
#include <vector>
#include <gbp/utils.h>
#include <gbp/Factor.h>
#include <gbp/Factorgraph.h>
#include <Eigen/Eigenvalues>
#include <algorithm>

extern Globals globals;

class Robot {
public:
    Robot(Simulator* sim,
                    std::vector<Eigen::VectorXd> waypoints,
                    float size,
                    Color color);
    ~Robot();


    std::mt19937 gen_normal = std::mt19937(globals.SEED);
    std::mt19937 gen_uniform = std::mt19937(globals.SEED);
    template<typename T>
    T random_number(std::string distribution, T param1, T param2){
        if (distribution=="normal") return std::normal_distribution<T>(param1, param2)(gen_normal);
        if (distribution=="uniform") return std::uniform_real_distribution<T>(param1, param2)(gen_uniform);
        return (T)0;
    }
    std::map<FGLayer, std::shared_ptr<FactorGraph>> fg {};
    int rid_ = 0;
    Simulator* sim_;
    std::vector<Eigen::VectorXd> waypoints_{};
    float robot_radius_ = 2.;
    Color color_ = DARKGREEN;
    Color future_color_ = Color{GRAY.r, GRAY.g, GRAY.b, 127};
    Color horizon_color_ = Color{color_.r, color_.g, color_.b, 127};    

    int num_varnodes_;
    std::vector<int> var_timesteps_list_{};
    Eigen::VectorXd initial_start_pose_;
    Eigen::VectorXd initial_horizon_pose_;     
    std::vector<int> connected_r_ids_{};
    bool found_ = false;

    float minval_ = 1e9;
    int minval_idx_ = -1;
    bool inactive_info_layer_ = false;
    Eigen::Vector2d best_value_point_{{0.,0.}};
    Eigen::Vector2d nearest_unexplored_point_{{0.,0.}};
    bool full_coverage_ = false;
    std::vector<Eigen::Vector2d> positions_{};
    float distance_travelled_ = 0;
    float height_3D_ = globals.ROBOT_RADIUS;

    Eigen::VectorXd real_pose_;
    Kdtree_formation* tree_;
    KdtreeInfoVars* kdtree_info_vars_;

    void updateCurrent();
    void updateHorizon();
    void updateGoalFactors();
    void updateInterrobotFactors();
    void createInterrobotFactors(std::shared_ptr<Robot> other_robot, bool symmetric_factors);
    void deleteInterrobotFactors(std::shared_ptr<Robot> other_robot, bool symmetric_factors);  
    void draw();

    std::pair<int,int> idx2grid(int idx);
    int grid2idx(std::pair<int,int> grid_idx);
    std::pair<int,int> pos2grid(Eigen::VectorXd pos);
    int pos2idx(Eigen::VectorXd pos);


    std::shared_ptr<Variable>& operator[] (const int& v_id){
        int n = this->fg[FGLayer::Planning]->variables_.size();
        int search_vid = ((n + v_id) % n + n) % n;

        auto it = this->fg[FGLayer::Planning]->variables_.begin();
        std::advance(it, search_vid);
        return it->second;
    }    

    friend class Simulator;
    
    std::vector<int> neighbours_{};

    template <typename Ta>
    bool contains(Ta& container, int& element){
        return std::find(container.begin(), container.end(), element)!=container.end();
    }


};

