/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <Eigen/Core>
#include <Eigen/Dense>
#include <raylib.h>
#include <vector>
#include <math.h>
#include <gbp/utils.h>
#include <gbp/key.h>
#include <gbp/Variable.h>
#include <gbp/Factor.h>

Variable::Variable(int v_id, int r_id, const Eigen::VectorXd& eta_prior, const Eigen::MatrixXd& lam_prior, Color color, float size, int n_dofs)
            : v_id_(v_id), r_id_(r_id), key_(Key(r_id, v_id)), eta_prior_(eta_prior), lam_prior_(lam_prior), color_(color), size_(size), n_dofs_(n_dofs) {
            }

Variable::Variable(int v_id, int r_id, const Eigen::VectorXd& mu_prior, const Eigen::VectorXd& sigma_prior_list, Color color, float size, int n_dofs)
            : v_id_(v_id), r_id_(r_id), key_(Key(r_id, v_id)), mu_(mu_prior), color_(color), size_(size), n_dofs_(n_dofs) {
                lam_prior_ = sigma_prior_list.cwiseProduct(sigma_prior_list).cwiseInverse().asDiagonal();
                if (!lam_prior_.allFinite()) lam_prior_.setZero();
                eta_prior_ = lam_prior_ * mu_prior;
                belief_ = Message(eta_prior_, lam_prior_, mu_prior);

            };

Variable::~Variable(){
    for (auto [fkey, fac] : factors_){
        delete_factor(fkey);
    }
}

bool Variable::update_belief(){
    // Collect messages from all other factors, begin by "collecting message from pose factor prior"
    eta_ = this->eta_prior_;
    lam_ = this->lam_prior_;

    for (auto& [f_key, msg] : this->inbox_) {
        auto [eta_msg, lam_msg, _] = msg;
        eta_ += eta_msg;
        lam_ += lam_msg;
    }
    double last_lam2 = lam_(0,0);
    // Update belief
    sigma_ = lam_.inverse();
    if (sigma_.allFinite()) {
        mu_ = sigma_ * eta_;
        this->is_valid = true; 
    } else{
        this->is_valid = false;
    }

    if (grid_idx>=std::pair<int,int>{0,0}){
        changed_ = (abs(mu_(2) - belief_.mu(2)) > 0);
    }
    this->belief_ = Message {this->eta_, this->lam_, this->mu_};

    // Create message to send to each factor that sent it stuff
    // msg is the aggregate of all OTHER factor messages (belief - last sent msg of that factor)
    for (auto [f_key, fac] : this->factors_) {
        this->outbox_[f_key] = belief_ - inbox_.at(f_key);
    }
    

    return  true;           
}

void Variable::add_factor(std::shared_ptr<Factor> fac){
    factors_[fac->key_] = fac;
    inbox_[fac->key_] = Message(n_dofs_);
    // outbox_[fac->key_] = Message(n_dofs_).setMu(belief_.mu);
    outbox_[fac->key_] = belief_;
}

void Variable::relax_covariance_and_update(float T){
    lam_prior_ = lam_ * pow(2, -1./T);
    eta_prior_ = eta_ * pow(2, -1./T);  
    
    // Update belief
    update_belief();
}

void Variable::delete_factor(Key fac_key){
    factors_.erase(fac_key);
    inbox_.erase(fac_key);
    // outbox_.erase(fac_key);            
}

void Variable::draw(bool filled){
    if (draw_fn_!=NULL){
        draw_fn_;
    } else {
        if (!is_valid) return;
        const int var_x = (mu_(0) );
        const int var_y = (mu_(1) );
        int size = (globals.DRAW_PATH && statetag_==StateTag::Intermediate) ? size_/4  : size_ ;
        if (globals.DRAW_PATH && statetag_==StateTag::Horizon) DrawText(TextFormat("%1i", r_id_), var_x + size, var_y, 10, color_);
        if (filled) DrawCircle(var_x, var_y, size, color_);
        DrawCircleLines(var_x, var_y, size, color_);
    }
}