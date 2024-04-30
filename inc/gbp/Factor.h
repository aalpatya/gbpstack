/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include "Simulator.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <gbp/utils.h>
#include <gbp/key.h>
#include <raylib.h>

extern Globals globals;

using Eigen::seqN;
using Eigen::seq;
using Eigen::last;

class Variable;

class Factor {
    public:
    int f_id_;
    int r_id_;
    Key key_;
    int other_rid_;
    int n_dofs_;
    Eigen::VectorXd factor_eta_;
    Eigen::MatrixXd factor_lam_;
    Eigen::VectorXd z_;
    Eigen::MatrixXd h_, J_;
    Eigen::VectorXd X_;
    Eigen::MatrixXd meas_model_lambda_;
    Mailbox inbox_, outbox_, last_outbox_;
    float sigma_;
    Color color_ = CLITERAL(Color){0,0,0,15};
    FactorType factor_type_ = NONE_FACTOR;
    float delta_jac=1e-8;
    float dt_ = 1.;
    float meas_func_strength_ = 1.f;
    bool initialised_ = false;
    bool linear_ = false;
    bool skip = false;
    double safety_distance_ = 0.;
    bool valid_ = true;
    virtual bool skip_factor(){
        skip = false;
        return skip;
    };
    Eigen::Vector2d origin_vec{{0.,0.}};
    Eigen::Vector2d desired_location_{{0.,0.}};
    Simulator* sim_;
    uint32_t ts = -1;

    std::vector<std::shared_ptr<Variable>> variables_{};
    std::pair<std::vector<Key>, std::map<Key, std::shared_ptr<Variable>>> list_variables_{};

    int64_t t1 = 0, t2 = 0;
    bool danger = false;

    Factor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
            float sigma, const Eigen::VectorXd& measurement, 
            int n_dofs=4);

    bool update_factor();
    Message applyDamping(Message& msg, Key vkey_out, double damping);
    Message marginalise_factor_dist(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, int var_idx);

    virtual Eigen::MatrixXd h_func_(const Eigen::VectorXd& X) = 0;
    virtual Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);
    virtual Eigen::VectorXd residual(){
        return z_ - h_;
    }
    Eigen::MatrixXd jacobianFirstOrder(const Eigen::VectorXd& X0);

    void draw();

    ~Factor();

};

class DynamicsFactor: public Factor {
    public:
    // Factor functions for 4D (state: [x,y,xdot,ydot]) dynamics

    DynamicsFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement, float dt);

    // Constant velocity model
    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);
    Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);

};

class InterrobotFactor: public Factor {
    public:
    float robot_radius_;

    InterrobotFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement, 
        float robot_size);

    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);
    Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);
    bool skip_factor();

};
class InterrobotGoalAvoidanceFactor: public Factor {
    public:
    float robot_radius_;

    InterrobotGoalAvoidanceFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement, float robot_size);

    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);
    Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);
    // bool skip_factor();

};
class GoalFactor: public Factor {
    public:

    GoalFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement);

    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);
    Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);
    bool skip_factor();

};

// Static environment factor
class ClosenessFactor: public Factor {
    public:

    ClosenessFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement);

    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);
    Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);

};