/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once

#include <vector>
#include <math.h>
#include <memory>

#include <gbp/utils.h>
#include <gbp/key.h>
// #include <gbp/Factor.h>
#include <raylib.h>
#include <Eigen/Core>
#include <Eigen/Dense>

class Factor;

class Variable {
    public:
        int v_id_;
        int r_id_;
        Key key_;
        Eigen::VectorXd eta_prior_;
        Eigen::MatrixXd lam_prior_;
        int n_dofs_;
        Eigen::VectorXd eta_;
        Eigen::MatrixXd lam_;
        Eigen::VectorXd mu_;
        Eigen::MatrixXd sigma_;
        Mailbox inbox_, outbox_;
        // Message belief_ = Message {this->eta_, this->lam_};
        Message belief_;
        bool is_valid = false;
        StateTag statetag_ = StateTag::Intermediate; // Intermediate state unless explicitly set
        std::pair<int, int> grid_idx{-1,-1};
        std::map<Key, std::shared_ptr<Factor>> factors_{};
        std::shared_ptr<Factor> unary_factor_;
        bool changed_ = true;

        float size_;
        Color color_;
        std::function<void()> draw_fn_ = NULL;
        uint32_t ts = -1;

        // Constructor
        Variable(int v_id, int r_id, const Eigen::VectorXd& eta_prior, const Eigen::MatrixXd& lam_prior, Color color, float size, int n_dofs=4);
        Variable(int v_id, int r_id, const Eigen::VectorXd& mu_prior, const Eigen::VectorXd& sigma_prior_list, Color color, float size, int n_dofs=4);
        ~Variable();

        bool update_belief();

        void relax_covariance_and_update(float T);

        // void add_factor(Key fac_key);
        void add_factor(std::shared_ptr<Factor> fac);

        void delete_factor(Key fac_key);

        virtual void draw(bool filled=true);

};