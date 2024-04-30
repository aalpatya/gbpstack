/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include <memory>

#include <gbp/utils.h>
#include <gbp/key.h>
#include <map>
#include <gbp/Factor.h>
#include <gbp/Variable.h>

#include <Eigen/Dense>
#include <Eigen/Core>

extern const int SCREEN_SZ;


class FactorGraph {
    public:
    std::map<Key, std::shared_ptr<Variable>> variables_{};
    std::map<Key, std::shared_ptr<Factor>> factors_{};
    int robot_id_;

    void factorIteration(bool internal = false, uint32_t simclock = UINT32_MAX);
    void variableIteration(bool internal = false, uint32_t simclock = UINT32_MAX);
    void iterate(int n);

    FactorGraph(int robot_id);

    std::shared_ptr<Variable>& getVar(const int& v_id){
        int n = variables_.size();
        int search_vid = ((n + v_id) % n + n) % n;

        auto it = variables_.begin();
        std::advance(it, search_vid);
        return it->second;
    }
    
    std::shared_ptr<Variable>& getVar(const Key& v_key){
        return variables_[v_key];
    }

};
