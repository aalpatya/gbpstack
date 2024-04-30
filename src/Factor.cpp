/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <gbp/utils.h>
#include <gbp/Factor.h>
#include <gbp/Variable.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <raylib.h>


Factor::Factor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement, 
        int n_dofs)
    : f_id_(f_id), r_id_(r_id), key_(r_id, f_id), variables_(variables), sigma_(sigma), z_(measurement), n_dofs_(n_dofs) {
        // Measurement model Lambda is 1/(sigma^2) Identity
        // TODO put variables_ filling in in main body of Factor(), rely only on vkey_ordered_list
        this->meas_model_lambda_ = Eigen::MatrixXd::Identity(z_.rows(), z_.rows()) / pow(sigma_,2.);
        // Initialise empty inbox and outbox
        Message zero_msg(n_dofs_);
        for (auto var : variables_) {
            inbox_[var->key_] = zero_msg;
            outbox_[var->key_] = zero_msg;
        }
        other_rid_=r_id_;

        // Initialise empty linearisation point
        X_ = Eigen::VectorXd::Zero(variables_.size()*n_dofs_);
    };

Factor::~Factor(){
    // for (auto var : variables_){
    //     var->delete_factor(key_);
    // }
}

void Factor::draw(){
    if ((factor_type_==INTERROBOT_FACTOR && globals.DRAW_FAC) || (factor_type_==DYNAMICS_FACTOR && globals.DRAW_PATH)){
        auto v_0 = *variables_.begin();
        auto v_1 = *std::next(variables_.begin());

        if (!v_0->is_valid || !v_1->is_valid) {return;};
        auto v_0_mu = v_0->mu_ ;
        auto v_1_mu = v_1->mu_ ;
        DrawLine(v_0_mu(0), v_0_mu(1), v_1_mu(0), v_1_mu(1), color_);
    }    
}

Eigen::MatrixXd h_func_(const Eigen::VectorXd& X){return X;};
Eigen::MatrixXd Factor::J_func_(const Eigen::VectorXd& X){return this->jacobianFirstOrder(X);};

Eigen::MatrixXd Factor::jacobianFirstOrder(const Eigen::VectorXd& X0){
    Eigen::MatrixXd h0 = h_func_(X0);    // Value at lin point
    Eigen::MatrixXd jac_out = Eigen::MatrixXd::Zero(h0.size(),X0.size());
    for (int i=0; i<X0.size(); i++){
        Eigen::VectorXd X_copy = X0;                                    // Copy of lin point
        X_copy(i) += delta_jac;                                         // Perturb by delta
        jac_out(Eigen::all, i) = (h_func_(X_copy) - h0) / delta_jac;    // Derivative (first order)
    }
    return jac_out;
};

bool Factor::update_factor(){
    // Update the state vector X for linearisation with incoming beliefs from variables
    int v=0;
    for (auto var : variables_){
        auto& [_, __, mu_belief] = this->inbox_[var->key_];
        X_(seqN(v*n_dofs_, n_dofs_)) = mu_belief;
        v++;
    }

    // This can't really happen, as when creating the factor we initialise it with the variable's state
    if (!X_.allFinite()){
        // print("ERROR at fac", f_id_, "Vars ", variables_[0]->v_id_, variables_[1]->v_id_, X_.transpose());

        for (auto var : variables_){
            this->outbox_[var->key_] = Message(var->n_dofs_);
        }        
        return false;
    }

    // We may need to skip computation of this factor, in which case send out a Zero Message
    if (this->skip_factor()){
        for (auto var : variables_){
            this->outbox_[var->key_] = Message(var->n_dofs_);
        }           
        return false;
    }
    
    // Calculate Factor Jacobian if non-linear factor, or doing for the first time.
    h_ = h_func_(X_);
    J_ = (this->linear_ && this->initialised_)? J_ : this->J_func_(X_);
    // Calculate linearised Factor Lambda and Eta
    factor_lam_ = J_.transpose() * meas_model_lambda_ * J_;
    factor_eta_ = (J_.transpose() * meas_model_lambda_) * (J_ * X_ + residual());
    this->initialised_ = true;

    //  Update appropriate region of factor_lam with belief - msg out of each variable
    int marginalisation_idx = 0;    // Used by the marginalisation function. Index of variables_ to marginalise
    for (auto var_out : variables_){
        // Initialise with factor values
        Eigen::VectorXd factor_eta = factor_eta_;     
        Eigen::MatrixXd factor_lam = factor_lam_;
        
        // Combine the factor with the (belief - previously_sent_message) from other variables apart from the receiving variable
        int idx = 0;
        for (auto var : variables_){
            if (var->key_ != var_out->key_) {
                auto [eta_belief, lam_belief, _] = inbox_[var->key_]; // from inbox
                factor_eta(seqN(idx, n_dofs_)) += eta_belief;
                factor_lam(seqN(idx, n_dofs_), seqN(idx, n_dofs_)) += lam_belief;
            }
            idx += n_dofs_;
        }
        
        // Marginalise n = [a,b] -> marignalise to send to variable a: n_a - L_ab L_bb^-1 n_b
        Message msg_out = marginalise_factor_dist(factor_eta, factor_lam, marginalisation_idx++);
        
        // /* DAMPING */
        // msg_out = applyDamping(msg_out, var_out->key_, globals.DAMPING);

        // Update the factor outbox with our new messages
        outbox_[var_out->key_] = msg_out;
    }

    return true;
};

Message Factor::applyDamping(Message& msg, Key vkey_out, double damping){
    /* DAMPING */
    if (last_outbox_.count(vkey_out)){
        auto& [last_eta, last_lambda, _] = last_outbox_.at(vkey_out);
        msg.eta = (1.f - damping) * msg.eta + damping * last_eta;
        msg.lambda = (1.f - damping) * msg.lambda + damping * last_lambda;    
    }
    last_outbox_.insert_or_assign(vkey_out, msg);
    return msg;    
}

Message Factor::marginalise_factor_dist(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, int var_idx){
    // Marginalisation only needed if factor is connected to >1 variables
    if (eta.size() == n_dofs_) return Message {eta, Lam};

    Eigen::VectorXd eta_a(n_dofs_), eta_b(eta.size()-n_dofs_);
    int marg_idx = var_idx * n_dofs_;                    // index at which our mariginalising variable begins
    eta_a = eta(seqN(marg_idx, n_dofs_));
    eta_b << eta(seq(0, marg_idx - 1)), eta(seq(marg_idx + n_dofs_, last));

    Eigen::MatrixXd lam_aa(n_dofs_, n_dofs_), lam_ab(n_dofs_, Lam.cols()-n_dofs_);
    Eigen::MatrixXd lam_ba(Lam.rows()-n_dofs_, n_dofs_), lam_bb(Lam.rows()-n_dofs_, Lam.cols()-n_dofs_);
    lam_aa << Lam(seqN(marg_idx, n_dofs_), seqN(marg_idx, n_dofs_));
    lam_ab << Lam(seqN(marg_idx, n_dofs_), seq(0, marg_idx - 1)), Lam(seqN(marg_idx, n_dofs_), seq(marg_idx + n_dofs_, last));
    lam_ba << Lam(seq(0, marg_idx - 1), seq(marg_idx, marg_idx + n_dofs_ - 1)), Lam(seq(marg_idx + n_dofs_, last), seqN(marg_idx, n_dofs_));
    lam_bb << Lam(seq(0, marg_idx - 1), seq(0, marg_idx - 1)), Lam(seq(0, marg_idx - 1), seq(marg_idx + n_dofs_, last)),
            Lam(seq(marg_idx + n_dofs_, last), seq(0, marg_idx - 1)), Lam(seq(marg_idx + n_dofs_, last), seq(marg_idx + n_dofs_, last));

    Eigen::MatrixXd lam_bb_inv = lam_bb.inverse();
    Message marginalised_msg(n_dofs_);
    marginalised_msg.eta = eta_a - lam_ab * lam_bb_inv * eta_b;
    marginalised_msg.lambda = lam_aa - lam_ab * lam_bb_inv * lam_ba;
    if (!marginalised_msg.lambda.allFinite()) marginalised_msg.setZero();

    return marginalised_msg;
};    

/********************************************************************************************/
/* Dynamics factor: constant-velocity model */

DynamicsFactor::DynamicsFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::VectorXd& measurement, 
    float dt)
    : Factor{f_id, r_id, variables, sigma, measurement}{ 
        color_ = BLACK;
        this->dt_ = dt; // number of timesteps
        this->factor_type_ = DYNAMICS_FACTOR;
        double sig2_inv = pow(sigma_, -2.);
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_dofs_/2,n_dofs_/2);
        Eigen::MatrixXd O = Eigen::MatrixXd::Zero(n_dofs_/2,n_dofs_/2);
        Eigen::MatrixXd Qc_inv = sig2_inv * I;

        Eigen::MatrixXd Qi_inv(n_dofs_, n_dofs_);
        Qi_inv << 12.*pow(dt_, -3.) * Qc_inv,   -6.*pow(dt_, -2.) * Qc_inv,
                  -6.*pow(dt_, -2.) * Qc_inv,   4./dt_ * Qc_inv;   

        this->meas_model_lambda_ = Qi_inv;        

        // Store Jacobian as it is linear
        this->linear_ = true;
        J_ = Eigen::MatrixXd::Zero(n_dofs_, n_dofs_*2);
        J_ << I, dt_*I, -1*I,    O,
             O,    I,    O, -1*I; 

    };

Eigen::MatrixXd DynamicsFactor::h_func_(const Eigen::VectorXd& X){
    return J_ * X;
}    
Eigen::MatrixXd DynamicsFactor::J_func_(const Eigen::VectorXd& X){
    return J_;
}

/********************************************************************************************/
/* Obstacle factor: for avoidance of other robots */

InterrobotFactor::InterrobotFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::VectorXd& measurement, 
    float robot_size)
    : Factor{f_id, r_id, variables, sigma, measurement}, robot_radius_(robot_size) {  
        this->factor_type_ = INTERROBOT_FACTOR;
        float eps = 0.2 * robot_radius_;
        this->safety_distance_ = 2*robot_radius_ + eps;
        this->delta_jac = 1e-2;
};

Eigen::MatrixXd InterrobotFactor::h_func_(const Eigen::VectorXd& X){
    // Each variable only has knowledge of itself and assumes other obstacle has same size.
    // This is ok because there will also be another factor from the other robot taking into account its size, and these factors will sum
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(),z_.cols());
    Eigen::VectorXd X_diff = X(seqN(0,n_dofs_/2)) - X(seqN(n_dofs_, n_dofs_/2));

    double r = X_diff.norm();
    if (r <= safety_distance_){
        this->skip = false;
        h(0) = 1.f*(1 - r/safety_distance_);
    }
    else {
        this->skip = true;
    }

    return h;
};

Eigen::MatrixXd InterrobotFactor::J_func_(const Eigen::VectorXd& X){
    // Each variable only has knowledge of itself and assumes other obstacle has same size.
    // This is ok because there will also be another factor from the other robot taking into account its size, and these factors will sum
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(z_.rows(), n_dofs_*2);
    Eigen::VectorXd X_diff = X(seqN(0,n_dofs_/2)) - X(seqN(n_dofs_, n_dofs_/2));
    double r = X_diff.norm();
    if (r <= safety_distance_){
        // J(0,Eigen::all) = -1.f/safety_distance_/r * (Eigen::VectorXd(n_dofs_*2) << X_diff, Eigen::VectorXd::Zero(n_dofs/2), -1. * X_diff, Eigen::VectorXd::Zero(n_dofs/2)).finished();
        J(0,seqN(0, n_dofs_/2)) = -1.f/safety_distance_/r * X_diff;
        J(0,seqN(n_dofs_, n_dofs_/2)) = 1.f/safety_distance_/r * X_diff;
    }
    return J;
};
bool InterrobotFactor::skip_factor(){
    this->skip = ( (X_(seqN(0,n_dofs_/2)) - X_(seqN(n_dofs_, n_dofs_/2))).squaredNorm() >= safety_distance_*safety_distance_ );
    return this->skip;
}

/********************************************************************************************/
ClosenessFactor::ClosenessFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::VectorXd& measurement)
    : Factor{f_id, r_id, variables, sigma, measurement, variables.front()->n_dofs_}{
        this->linear_ = true;
        this->factor_type_ = CLOSENESS_FACTOR;
};
Eigen::MatrixXd ClosenessFactor::h_func_(const Eigen::VectorXd& X){
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(),1);
    auto I = Eigen::MatrixXd::Identity(z_.rows(), n_dofs_);
    Eigen::MatrixXd J(z_.rows(), 2*n_dofs_); J << I, -I;
    h = J * X;
    return h;
};
Eigen::MatrixXd ClosenessFactor::J_func_(const Eigen::VectorXd& X){
    auto I = Eigen::MatrixXd::Identity(z_.rows(), n_dofs_);
    Eigen::MatrixXd J(z_.rows(), 2*n_dofs_); J << I, -I;
    return J;
};

GoalFactor::GoalFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::VectorXd& measurement)
    : Factor{f_id, r_id, variables, sigma, measurement, 2}{
        desired_location_ = Eigen::Vector2d::Zero(2);
};
Eigen::MatrixXd GoalFactor::h_func_(const Eigen::VectorXd& X){
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(),1);
    
    h = meas_func_strength_ * (X - desired_location_);
    return h;
};
Eigen::MatrixXd GoalFactor::J_func_(const Eigen::VectorXd& X){
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(2, 2);
    return meas_func_strength_ * J;
};
bool GoalFactor::skip_factor(){
    return skip;
};

InterrobotGoalAvoidanceFactor::InterrobotGoalAvoidanceFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::VectorXd& measurement, float robot_size)
    : Factor{f_id, r_id, variables, sigma, measurement, variables.front()->n_dofs_}, robot_radius_(robot_size) {  
        this->factor_type_ = INTERROBOT_FACTOR;
        float eps = 0.2 * robot_radius_;
        this->safety_distance_ = globals.WORLD_SZ/globals.gn;//10*robot_radius_ + eps;
        // this->safety_distance_ = globals.MAPPING_RADIUS;//10*robot_radius_ + eps;
};

Eigen::MatrixXd InterrobotGoalAvoidanceFactor::h_func_(const Eigen::VectorXd& X){
    // Each variable only has knowledge of itself and assumes other obstacle has same size.
    // This is ok because there will also be another factor from the other robot taking into account its size, and these factors will sum
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(),z_.cols());
    // Eigen::VectorXd X_diff = X(seqN(0,2)) - X(seqN(2, 2)) + 1e-4*Eigen::VectorXd::Ones(2);
    Eigen::VectorXd X_diff = X(seqN(0,2)) - X(seqN(2, 2)) + 1e-4*Eigen::VectorXd{{(double)r_id_, (double)other_rid_}};

    double r = X_diff.norm();
    if (r < safety_distance_){
        h(0) = 1.f*(1 - r/safety_distance_);
    }
    return h;
};

Eigen::MatrixXd InterrobotGoalAvoidanceFactor::J_func_(const Eigen::VectorXd& X){
    // Each variable only has knowledge of itself and assumes other obstacle has same size.
    // This is ok because there will also be another factor from the other robot taking into account its size, and these factors will sum
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(z_.rows(), 2*2);
    // Eigen::VectorXd X_diff = X(seqN(0,2)) - X(seqN(2, 2)) + 1e-4*Eigen::VectorXd::Ones(2);
    Eigen::VectorXd X_diff = X(seqN(0,2)) - X(seqN(2, 2)) + 1e-4*Eigen::VectorXd{{(double)r_id_, (double)other_rid_}};
    
    double r = X_diff.norm();
    if (r < safety_distance_){
        J(0,seqN(0, 2)) = -1.f/safety_distance_/r * X_diff;
        J(0,seqN(2, n_dofs_)) = 1.f/safety_distance_/r * X_diff;
    }
    return J;
};