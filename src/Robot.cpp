/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <iostream>

#include <gbp/Robot.h>
#include <omp.h>
#pragma omp declare reduction (merge : std::vector<std::tuple<int, float, float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (mergemin : std::vector<std::tuple<int, float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

Robot::Robot(Simulator* sim,
                    std::vector<Eigen::VectorXd> waypoints,
                    float size,
                    Color color) : 
                    sim_(sim), rid_(sim->next_rid_),
                    waypoints_(waypoints),
                    robot_radius_(size), color_(color) {

    /***********************************************************/
    // Define Factor Graph Layers here.
    fg[FGLayer::Planning] = std::make_shared<FactorGraph>(rid_);
    fg[FGLayer::Information] = std::make_shared<FactorGraph>(rid_);
    fg[FGLayer::Goal] = std::make_shared<FactorGraph>(rid_);

    initial_start_pose_ = real_pose_ = waypoints_[0];
    initial_horizon_pose_ = waypoints_[min(1,waypoints_.size()-1)];

    // Variables representing the planned path are at timesteps which increase in spacing.
    // eg. (so that a span of 10 timesteps as a planning horizon can be represented by much fewer variables)    
    var_timesteps_list_ = get_variable_list(globals.T_HORIZON / globals.T0, globals.LOOKAHEAD_MULTIPLE);
    num_varnodes_ = var_timesteps_list_.size();

    /***************************************************************************/
    /* Create Variables with fixed pose priors on start and horizon variables. */
    Color var_color; double sigma; double sigma_inv_sq;
    int n = globals.N_DOFS;
    Eigen::VectorXd mu(n); Eigen::VectorXd sigma_list(n); 
    for (int i = 0; i < num_varnodes_; i++){
        // Determine Variable type
        auto var_type = (i==0) ? StateTag::Current :  (i==num_varnodes_-1) ? StateTag::Horizon : StateTag::Intermediate;
        
        // Set initial mu and covariance of variable.
        mu = this->initial_start_pose_ + (initial_horizon_pose_-initial_start_pose_) * (float)(var_timesteps_list_[i]/(float)var_timesteps_list_.back());
        sigma = (var_type==StateTag::Current) ? globals.SIGMA_POSE_FIXED : (var_type==StateTag::Horizon) ? globals.SIGMA_POSE_END : 0.;
        for (int ii=0; ii<n; ii++) sigma_list(ii) = sigma;
        // Make covariance stronger on the velocity states in the horizon variable.
        if (var_type==StateTag::Horizon) sigma_list(seqN(n/2, n/2)).setConstant(globals.SIGMA_POSE_FIXED);
        
        // Create the variable
        auto variable = std::make_shared<Variable>(sim->next_vid_++, rid_, mu, sigma_list, var_color, robot_radius_, n);
        variable->color_ = (var_type==StateTag::Current) ? color_ : (var_type==StateTag::Horizon) ? horizon_color_ : future_color_;
        variable->statetag_ = var_type;

        // Add variable to robot's appropriate factor graph layer
        fg[FGLayer::Planning]->variables_[variable->key_] = variable;
    }

    /***************************************************************************/
    /* Create Dynamics factors */
    auto it1 = fg[FGLayer::Planning]->variables_.begin(); auto it2 = std::next(it1);
    for (int i = 0; i < num_varnodes_ - 1; i++)
    {
        float delta_t = globals.T0 * (this->var_timesteps_list_[i + 1] - this->var_timesteps_list_[i]);
        std::vector<std::shared_ptr<Variable>> variables {it1++->second, it2++->second};
        auto factor = std::make_shared<DynamicsFactor>(sim->next_fid_++, rid_, variables, globals.SIGMA_FACTOR_DYNAMICS, Eigen::VectorXd::Zero(globals.N_DOFS), delta_t);
        for (auto var : factor->variables_) var->add_factor(factor);
        fg[FGLayer::Planning]->factors_[factor->key_] = factor;
    }

    /***************************************************************************/
    /*****************************************************************************/
    // GBP LAYER (INFORMATION LEVEL LAYER)
    std::vector<Eigen::Vector2d> info_var_points{};
    for (int ix=0; ix<globals.gn*globals.gn; ix++){
        Eigen::VectorXd mu = Eigen::VectorXd{{(ix%globals.gn + 0.5)*(sim_->groundTruthImg.width/(double)globals.gn) - sim_->groundTruthImg.width/2.,
                                              (ix/globals.gn + 0.5)*(sim_->groundTruthImg.height/(double)globals.gn) - sim_->groundTruthImg.height/2.,
                                              1.,
                                              0.}};
        Eigen::VectorXd sigma_list {{0.,0.,0.,0.}};    // Covariance on value and coverage
        auto variable_info = std::make_shared<Variable>(sim_->next_vid_++, rid_, mu, sigma_list, DARKGREEN, robot_radius_, mu.size());
        variable_info->grid_idx = {ix%globals.gn, ix/globals.gn};
        fg[FGLayer::Information]->variables_[variable_info->key_] = variable_info;
        variable_info->update_belief();
        info_var_points.push_back(mu({0,1}));
    }
    kdtree_info_vars_ = new KdtreeInfoVars(info_var_points);
    /*****************************************************************************/
    // GBP LAYER (GOAL LEVEL LAYER)
    {
        Eigen::VectorXd mu{{0.,0.}};
        Eigen::VectorXd sigma_list {{0., 0.}};
        auto variable = std::make_shared<Variable>(sim->next_vid_++, rid_, mu, sigma_list, RED, robot_radius_, mu.size());
        fg[FGLayer::Goal]->variables_[variable->key_] = variable;
        variable->update_belief();

        // Add factors
        std::vector<std::shared_ptr<Variable>> variables{variable};
        Eigen::VectorXd z{{0.,0.}}; std::vector<std::shared_ptr<GoalFactor>> created_factors;

        // EXPLORATION FACTOR
        created_factors.push_back(std::make_shared<GoalFactor>(sim->next_fid_++, rid_, variables, globals.SIGMA_FACTOR_COVERAGE_GOAL, z));

        // SIGNAL FACTOR
        created_factors.push_back(std::make_shared<GoalFactor>(sim->next_fid_++, rid_, variables, globals.SIGMA_FACTOR_SEEKING_GOAL, z));

        for (auto factor : created_factors){
            for (auto var : factor->variables_) {var->add_factor(factor); var->outbox_[factor->key_].setZero();}
            fg[FGLayer::Goal]->factors_[factor->key_] = factor;
        }
    }
};

/***************************************************************************************************/
/* Destructor */
Robot::~Robot(){
}

/***************************************************************************************************/
/* Change the prior of the start (current) variable */
void Robot::updateCurrent(){
    // Store the plan
    Eigen::VectorXd delta_plan = (globals.PLANNING) * ((*this)[1]->mu_ - (*this)[0]->mu_) * globals.TIMESTEP / globals.T0;

    // Real pose update
    real_pose_ = real_pose_ + delta_plan;
    distance_travelled_ += delta_plan.norm();

    // Move plan: move plan current state by plan increment
    sim_->change_variable_prior(fg[FGLayer::Planning]->getVar(0), fg[FGLayer::Planning]->getVar(0)->mu_ + delta_plan);

};
/***************************************************************************************************/
/* Change the prior of the horizon state */
void Robot::updateHorizon(){

    auto horizon = fg[FGLayer::Planning]->getVar(-1);                                                           // get horizon state
    Eigen::VectorXd dist_horz_to_goal = fg[FGLayer::Goal]->getVar(0)->mu_ - horizon->mu_({0,1});                // distance from horizon to current goal
    Eigen::VectorXd new_vel = dist_horz_to_goal.normalized() * globals.MAX_SPEED;// cap horizon velocity in direction of goal
    if (dist_horz_to_goal.norm()<0.1){
        new_vel = Eigen::VectorXd::Random(2);
    }
    
    // Deal with wall boundaries.
    float b_min = -globals.WORLD_SZ/2.f, b_max = globals.WORLD_SZ/2.f;
    Eigen::VectorXd new_pos = horizon->mu_({0,1}) + new_vel*globals.TIMESTEP;
    // Update horizon state with new pos and vel
    horizon->mu_ << new_pos, new_vel;
    sim_->change_variable_prior(horizon, horizon->mu_);
}


void Robot::updateGoalFactors(){
    /**********************************************/
    /* COLLECT INFORMATION FROM INFO VARIABLES */
    auto current = fg[FGLayer::Planning]->getVar(0);                                                // get current state
    Eigen::Vector2d X_0 = current->mu_({0,1});                                                      // Current position
    float threshold = 10./255.;                                                                     // Threshold for signal

    /* COLLECT INFORMATION FROM INFO LAYER */

    // Set the information variables within comms range of other robots to be "active"
    std::vector<nanoflann::ResultItem<size_t, double>>  nearby_vars = kdtree_info_vars_->search(X_0, pow(globals.COMMUNICATION_RADIUS,2.));
    #pragma omp parallel for
    for (int idx=0; idx<nearby_vars.size(); idx++){
        int i = nearby_vars[idx].first;    
        auto v_it = fg[FGLayer::Information]->variables_.begin(); std::advance(v_it, i);
        auto [v_key, info_var] = *v_it;
        for (auto& [_, f] : info_var->factors_){
            f->ts = sim_->clock_;
        }
    }

    // Get the best seen signal location for Signal Factor, and the nearest unexplored location for Exploration Factor
    std::vector<std::tuple<int, float, float>> scores{}; // idx, coverage, distance
    std::vector<std::tuple<int, float>> minscores{}; // idx, coverage, distance
    std::vector<nanoflann::ResultItem<size_t, double>>  results = kdtree_info_vars_->search(X_0, pow(2.*globals.WORLD_SZ, 2.));
    #pragma omp parallel for reduction(merge: scores) reduction(mergemin: minscores)
    for (int idx=0; idx<results.size(); idx++){
        nanoflann::ResultItem<size_t, double> result = results[idx];
        int i = result.first;    
        auto v_it = fg[FGLayer::Information]->variables_.begin(); std::advance(v_it, i);
        auto [v_key, info_var] = *v_it;
        double coverage = info_var->mu_(3);                     // Coverage at info variable
        float distSq = result.second;
        scores.push_back({i, coverage, distSq});
        minscores.push_back({i, info_var->mu_(2)});
    }

    /* ****************************************************** */
    /* UPDATE FACTORS IN GOAL LAYER*/
    // "it" is iterator to the factors. it->first gives the factor's key, it->second gives the factor itself.
    auto it = fg[FGLayer::Goal]->factors_.begin();

    // EXPLORATION FACTOR: Draws goal variable towards nearest unexplored location
    if (full_coverage_){
        if ((nearest_unexplored_point_ - X_0).norm()<globals.MAPPING_RADIUS){
            int random = rand()%(globals.gn*globals.gn);
            nearest_unexplored_point_ = fg[FGLayer::Information]->getVar(random)->mu_({0,1});
        }
    } else {
        if (!scores.empty()){
            auto best = std::min_element(scores.begin(), scores.end(), [](const std::tuple<int, float, float>& a, const std::tuple<int, float, float>& b){return (std::get<1>(a)==std::get<1>(b)) ? (std::get<2>(a) < std::get<2>(b)) : (std::get<1>(a) < std::get<1>(b));});
            nearest_unexplored_point_ = fg[FGLayer::Information]->getVar(std::get<0>(*best))->mu_({0,1});   
        } else {
            print("EMPTY", results.size(), X_0.transpose(), globals.WORLD_SZ);
            nearest_unexplored_point_ = Eigen::VectorXd{{0.,0.}};
        }
    }
    // Set the 'desired_location' of the factor to the nearest unexplored point (this factor encourages the goal variable to move towards its desired_location)
    it++->second->desired_location_ = nearest_unexplored_point_;

    // SIGNAL FACTOR: Draws goal towards current best signal value location
    if (!minscores.empty()){
        auto best = std::min_element(minscores.begin(), minscores.end(), [](const std::tuple<int, float>& a, const std::tuple<int, float>& b){return std::get<1>(a) < std::get<1>(b);});
        best_value_point_ = fg[FGLayer::Information]->getVar(std::get<0>(*best))->mu_({0,1});
        minval_ = std::get<1>(*best);
        minval_idx_ = std::get<0>(*best);
    } else {
        best_value_point_ = Eigen::VectorXd{{0.,0.}};
    }
    // Set the 'desired_location' of the factor to the best valued point
    // (this factor encourages the goal variable to move towards its desired_location)
    // Weight the measurement function (or the factor really) by the current minimum found value.
    it->second->desired_location_ = best_value_point_;
    it++->second->meas_func_strength_ = 1.f - minval_;
    /* ****************************************************** */

    // Bookkeeping
    found_ = (globals.MODE==0 && (minval_<threshold && sim_->groundTruthMins[minval_idx_]<threshold));

    if (found_){
        // for (auto rid : connected_r_ids_){
        //     sim_->robots_[rid]->found_ = true;
        // }
    }


};

/***************************************************************************************************/
/* Check if other robots are in the vicinity. Create factors if they don't exist. Delete factors for faraway robots */
void Robot::updateInterrobotFactors(){
    // bool symmetric_factors = true; // Toggle whether symmetric factors to be created (identical factors between 0->1 and 1->0)
    bool symmetric_factors = false; // Toggle whether symmetric factors to be created (identical factors between 0->1 and 1->0)
    // Search through currently connected rids. If any are not in neighbours, delete interrobot factors.
    for (auto rid : connected_r_ids_){
        if (!contains(neighbours_, rid)){
            deleteInterrobotFactors(sim_->robots_[rid], symmetric_factors);
        };
    }
    // Search through neighbours. If any are not in currently connected rids, create interrobot factors.
    for (auto rid : neighbours_){
        if (!contains(connected_r_ids_, rid)){
            createInterrobotFactors(sim_->robots_[rid], symmetric_factors);
        };
    }
}

/***************************************************************************************************/
/* Create factor between this robot and another robot */
void Robot::createInterrobotFactors(std::shared_ptr<Robot> other_robot, bool symmetric_factors)
{
    /*********************************************************************************************************/
    // Planning Layer
    /*********************************************************************************************************/
    // Create Interrobot factors for all timesteps excluding current state but including horizon
    for (int i = 1; i < num_varnodes_; i++){
        // GET VARIABLES
        std::vector<std::shared_ptr<Variable>> variables{fg[FGLayer::Planning]->getVar(i), 
                                                        other_robot->fg[FGLayer::Planning]->getVar(i)};
        // Covariance on states becomes weaker for states further into the future
        float sigma = globals.SIGMA_FACTOR_INTERROBOT;

        // CREATE FACTOR
        Eigen::VectorXd z = Eigen::VectorXd::Zero(variables.front()->n_dofs_);
        auto factor = std::make_shared<InterrobotFactor>(sim_->next_fid_++, this->rid_, variables, sigma, z, 0.5*(this->robot_radius_ + other_robot->robot_radius_));
        factor->other_rid_ = other_robot->rid_;
        // ADD FACTOR TO VARIABLE
        for (auto var : factor->variables_) var->add_factor(factor);
        // UPDATE LIST OF FACTORS
        this->fg[FGLayer::Planning]->factors_[factor->key_] = factor;
    }

    /*********************************************************************************************************/
    // Information Layer
    /*********************************************************************************************************/
    for (int i=0; i<(globals.gn*globals.gn); i++){
        // GET VARIABLES
        std::vector<std::shared_ptr<Variable>> variables{fg[FGLayer::Information]->getVar(i), 
                                                        other_robot->fg[FGLayer::Information]->getVar(i)};
        // CREATE FACTOR
        Eigen::VectorXd z = Eigen::VectorXd::Zero(variables.front()->n_dofs_);
        float sigma = globals.SIGMA_FACTOR_CONSENSUS;
        auto factor = std::make_shared<ClosenessFactor>(sim_->next_fid_++, this->rid_, variables, sigma, z);
        factor->other_rid_ = other_robot->rid_;
        factor->ts = globals.SAMPLE_TIMESTEP * (sim_->clock_/(int)globals.SAMPLE_TIMESTEP + 1);
        // ADD FACTOR TO VARIABLE
        for (auto var : factor->variables_) {
            var->add_factor(factor);
        }
        // UPDATE LIST OF FACTORS
        this->fg[FGLayer::Information]->factors_[factor->key_] = factor;
    }
    /*********************************************************************************************************/
    // Goal Layer
    /*********************************************************************************************************/
    {
        // GET VARIABLES
        std::vector<std::shared_ptr<Variable>> variables{fg[FGLayer::Goal]->getVar(0), 
                                                        other_robot->fg[FGLayer::Goal]->getVar(0)};
        
        // CREATE FACTOR
        Eigen::VectorXd z = Eigen::VectorXd::Zero(variables.front()->n_dofs_);
        float sigma = globals.SIGMA_FACTOR_INTERROBOT_GOAL;
        auto factor = std::make_shared<InterrobotGoalAvoidanceFactor>(sim_->next_fid_++, this->rid_, variables, sigma, z, 0.5*(this->robot_radius_ + other_robot->robot_radius_));
        factor->other_rid_ = other_robot->rid_;
        // ADD FACTOR TO VARIABLE
        for (auto var : factor->variables_) var->add_factor(factor);
        // UPDATE LIST OF FACTORS
        this->fg[FGLayer::Goal]->factors_[factor->key_] = factor;
    }

    this->connected_r_ids_.push_back(other_robot->rid_);
    if (!symmetric_factors) other_robot->connected_r_ids_.push_back(rid_);
};

/***************************************************************************************************/
/* Delete interrobot factors between the two robots */
void Robot::deleteInterrobotFactors(std::shared_ptr<Robot> other_robot, bool symmetric_factors)
{
    /*********************************************************************************************************/
    // Planning Layer
    /*********************************************************************************************************/
    {
        std::vector<Key> facs_to_delete{};
        for (auto& [f_key, fac] : this->fg[FGLayer::Planning]->factors_){
            if (fac->other_rid_ != other_robot->rid_) continue;

            // Only get here if factor is connected to a variable in the other_robot
            for (auto& var : fac->variables_){ 
                var->delete_factor(f_key);
                facs_to_delete.push_back(f_key);
            }
        }
        for (auto f_key : facs_to_delete) this->fg[FGLayer::Planning]->factors_.erase(f_key);
    }
    /*********************************************************************************************************/
    // Information Layer
    /*********************************************************************************************************/
    {
        std::vector<Key> facs_to_delete{};
        for (auto& [f_key, fac] : this->fg[FGLayer::Information]->factors_){
            if (fac->other_rid_ != other_robot->rid_) continue;

            // Only get here if factor is connected to a variable in the other_robot
            for (auto& var : fac->variables_){ 
                var->delete_factor(f_key);
                facs_to_delete.push_back(f_key);
            }
        }
        for (auto f_key : facs_to_delete) this->fg[FGLayer::Information]->factors_.erase(f_key);
    }
    /*********************************************************************************************************/
    // Goal Layer
    /*********************************************************************************************************/
    {
        std::vector<Key> facs_to_delete{};
        for (auto& [f_key, fac] : this->fg[FGLayer::Goal]->factors_){
             if (fac->other_rid_ != other_robot->rid_) continue;

            // Only get here if factor is connected to a variable in the other_robot
            for (auto& var : fac->variables_){ 
                var->delete_factor(f_key);
                facs_to_delete.push_back(f_key);
            }
        }
        for (auto f_key : facs_to_delete) this->fg[FGLayer::Goal]->factors_.erase(f_key);
    }

    // Remove other robot from current robot's connected rids
    auto it = std::find(connected_r_ids_.begin(), connected_r_ids_.end(), other_robot->rid_);
    if (it != connected_r_ids_.end()){
        connected_r_ids_.erase(it);
    }
    if (!symmetric_factors) {
        auto it = std::find(other_robot->connected_r_ids_.begin(), other_robot->connected_r_ids_.end(), rid_);
        if (it != other_robot->connected_r_ids_.end()){
            // other_robot->connected_r_ids_.erase(it);
        }
    }


};

void Robot::draw(){
    // Camera raylib considers y axis as vertical, we use -z axis as vertical
    Color col = (inactive_info_layer_) ? GRAY : color_;
    if (globals.DRAW_PATH){
        // Planning layer Variables 
        int v=0;
        static int debug = 0;
        for (auto [vid, variable] : fg[FGLayer::Planning]->variables_){
            if (v++==0 || !variable->is_valid) continue;
            Eigen::VectorXd startpos = variable->mu_({0,1}) ;
            DrawSphere(Vector3{(float)startpos(0), height_3D_, (float)startpos(1)}, 0.5*robot_radius_, ColorAlpha(col, 0.5));
        }
    }     

    // Information layer
    if (globals.DRAW_INFO){ 
        float drawgrid_sz = 5; Eigen::VectorXd drawgrid00 = fg[FGLayer::Planning]->getVar(0)->mu_({0,1});
        float square_width = drawgrid_sz/(float)globals.gn;
        float y_offset = 1.5*drawgrid_sz;
        DrawCylinder(Vector3{(float)drawgrid00(0), height_3D_, (float)drawgrid00(1)}, 0.1f, 0.1f, y_offset, 4, col);
        for (auto [vkey, var] : fg[FGLayer::Information]->variables_){
            Eigen::VectorXd information_mu = var->mu_({2,3});
            float val = min(1., max(0., information_mu(0)));
            float sat = min(1., max(0., information_mu(1)));
            if (inactive_info_layer_) sat = 0.;
            DrawCubeV(Vector3{(float)drawgrid00(0)-drawgrid_sz/2.f+(square_width)*(var->grid_idx.first)+(square_width/2.f),
                                height_3D_ + y_offset + drawgrid_sz/2.f-(square_width)*(var->grid_idx.second)+(square_width/2.f),
                                (float)drawgrid00(1)},
                        Vector3{square_width, square_width, 0.25}, ColorFromHSV(ColorToHSV(col).x, sat, val));
        }
        DrawCubeWiresV(Vector3{(float)drawgrid00(0),
                                height_3D_ + y_offset + square_width,
                                (float)drawgrid00(1)},
                        Vector3{drawgrid_sz, drawgrid_sz, 0.25}, BLACK);
    }

    // Real pose
    DrawModel(sim_->graphics->robotModel_, Vector3{(float)real_pose_(0), height_3D_, (float)real_pose_(1)}, robot_radius_, col);
    if (globals.DRAW_GOAL){
        DrawCubeV(Vector3{(float)fg[FGLayer::Goal]->getVar(0)->mu_(0), height_3D_, (float)fg[FGLayer::Goal]->getVar(0)->mu_(1)}, Vector3{1.f*robot_radius_,1.f*robot_radius_,1.f*robot_radius_}, col);
    }
    
}

std::pair<int,int> Robot::idx2grid(int idx){
    return std::pair<int,int>{idx % globals.gn, idx / globals.gn};
}
int Robot::grid2idx(std::pair<int,int> grid_idx){
    return grid_idx.second * globals.gn + grid_idx.first;
}
std::pair<int,int> Robot::pos2grid(Eigen::VectorXd pos){
    return {(int)(floor((pos(0)+sim_->groundTruthImg.width/2)/(sim_->groundTruthImg.width/globals.gn))),
                                   (int)(floor((pos(1)+sim_->groundTruthImg.width/2)/(sim_->groundTruthImg.width/globals.gn)))};
}
int Robot::pos2idx(Eigen::VectorXd pos){
    return grid2idx(pos2grid(pos));
};


