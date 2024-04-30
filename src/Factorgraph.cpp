/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <gbp/Factorgraph.h>

// This is for the FactorGraph class.
// The robot class uses the FactorGraph class.

FactorGraph::FactorGraph(int robot_id) : robot_id_(robot_id){};

void FactorGraph::factorIteration(bool internal, uint32_t simclock){
#pragma omp parallel for    
    for (int f_idx=0; f_idx<factors_.size(); f_idx++){
        auto f_it = factors_.begin(); std::advance(f_it, f_idx);
        auto [f_key, fac] = *f_it;
        // Read message from each connected variable
        if (simclock!=UINT32_MAX && fac->ts != simclock) continue;

        for (auto var : fac->variables_){
            // If we are only doing internal (not interrobot) message passing, don't read msg from external node
            if (internal && var->key_.robot_id_!=robot_id_) continue;
            fac->inbox_[var->key_] = var->outbox_.at(f_key);
            if (simclock!=UINT32_MAX && fac->ts == simclock){
                var->ts = fac->ts;
            }
        }
        // Calculate factor potential and create outgoing messages
        fac->update_factor();
    };
};

void FactorGraph::variableIteration(bool internal, uint32_t simclock){
#pragma omp parallel for    
    for (int v_idx=0; v_idx<variables_.size(); v_idx++){
        auto v_it = variables_.begin(); std::advance(v_it, v_idx);
        auto [v_key, var] = *v_it;
        if (simclock!=UINT32_MAX && var->ts != simclock) var->update_belief();
        // Read message from each connected factor
        for (auto [f_key, fac] : var->factors_){
            // If we are only doing external (not interrobot) message passing, don't read msg from external node
            if (internal && f_key.robot_id_!=robot_id_) continue;
            var->inbox_[f_key] = fac->outbox_.at(v_key);
        }

        // Update variable belief and create outgoing messages
        var->update_belief();
    };
;}

void FactorGraph::iterate(int n){
    // For all factors in this factorgraph
    factorIteration();
    variableIteration();
};
