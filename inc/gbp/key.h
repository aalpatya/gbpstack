/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once

struct Key {
  int robot_id_;
  int node_id_;
  bool valid_;
  
  Key(int graph_id, int node_id, bool valid=true)
      : robot_id_(graph_id), node_id_(node_id), valid_(valid) {}

  friend bool operator== (const Key &key1, const Key &key2) {
    return key1.robot_id_ == key2.robot_id_ && key1.node_id_ == key2.node_id_;
  }
  friend bool operator!= (const Key &key1, const Key &key2) {
    return !(key1.robot_id_ == key2.robot_id_ && key1.node_id_ == key2.node_id_);
  }

  friend bool operator< (const Key &key1, const Key &key2) {
    return (key1.robot_id_ < key2.robot_id_) ||
           (key1.robot_id_ == key2.robot_id_ && key1.node_id_ < key2.node_id_);
  }
};
