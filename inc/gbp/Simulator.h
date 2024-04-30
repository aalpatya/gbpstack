/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once

#include <map>
#include <memory>

#include <gbp/utils.h>
#include <gbp/Variable.h>
#include <nanoflann.h>

#include <raylib.h>
#include <rlights.h>
#include <nanoflann.h>
#include <gbp/KDTreeVectorOfVectorsAdaptor.h>
#include <random>

class Robot;
class Graphics;
class Kdtree;
class Kdtree_formation;
class Kdtree_obstacle;

class Simulator {
public:
    Simulator(FormationName formation_name);
    FormationName formation_name_;
    Graphics* graphics;
    Kdtree* kdtree_;
    Kdtree_formation* kdtree_formation_;
    Kdtree_obstacle* kdtree_obstacle_;
    Image formation_img;
    Image obs_img;
    std::vector<Eigen::Vector2i> zlist{};
    Eigen::MatrixXd sdf;
    Eigen::MatrixXd obstacleSDF;
    Texture2D texture;
    Texture2D texture_img;
    std::vector<float> groundTruthMins{};
    std::vector<int> fail_robots_{};
    bool source_found_ = false;
    bool full_coverage_ = false;
    std::vector<int> dropout_rids_{};


    // RANDOM NUMBER GENERATOR. Usage: random_number("normal", mean, sigma) or random_number("uniform", lower, upper)
    
    std::mt19937 gen_normal = std::mt19937(globals.SEED);
    std::mt19937 gen_uniform = std::mt19937(globals.SEED);
    template<typename T>
    T random_number(std::string distribution, T param1, T param2){
        if (distribution=="normal") return std::normal_distribution<T>(param1, param2)(gen_normal);
        if (distribution=="uniform") return std::uniform_real_distribution<T>(param1, param2)(gen_uniform);
        return (T)0;
    }

    // Define the camera to look into our 3d world
    Camera3D camera3d = { 0 };   
    std::vector<Vector3> camera_positions_{};
    std::vector<Vector3> camera_ups_{};
    std::vector<Vector3> camera_targets_{};
    int camera_idx_ = 0;
    uint32_t camera_clock_=0;
    std::vector<float> rms_vector_{};   
    std::vector<float> coverage_vector_{};   

    // Image img_char;
    // Texture2D tex = LoadTextureFromImage(img_base);
    Image groundTruthImg;
    Color* groundTruthField;
    Color* obstacleColorArray;
    Color* p_colors_LEFT;
    Color* p_colors_RIGHT;

    ~Simulator();
    Ray ray_{};
    int next_rid_ = 0;
    int next_vid_ = 0;
    int next_fid_ = 0;
    uint32_t clock_ = 0;
    uint32_t time_finished_ = UINT32_MAX;
    uint32_t iterations_ = 0;
    std::map<int, std::shared_ptr<Robot>> robots_;
    // Background background_ = Background(BACKGROUND_NONE);
    // Background background_ = Background(BACKGROUND_CLUTTER);
    // Eigen::MatrixXd* p_sdf_matrix_ = &background_.sdf_matrix_;
    bool do_update_ = true; // Whether or not to create new robots

    void update_camera();
    
    int addRobot(std::shared_ptr<Robot>);

    void updateSetup();

    void timestep();

    void logMetrics();

    void change_variable_prior(std::shared_ptr<Variable> p_var, const Eigen::VectorXd& new_mu);

    void draw();


    void iterate_gbp(int n_iters, FGLayer layer = FGLayer::Planning, bool internal = false);

    friend class Robot;
    friend class Factor;

    int clicked_id_ = -1;
    std::set<std::pair<int,int>> collisions_obstacles_{};
    std::set<std::pair<int,int>> collisions_robots_{};


};

class Graphics {
public:
    Graphics(Simulator* sim);
    ~Graphics();
    void updateShader();

    Simulator* sim_;
    Model robotModel_;
    Model earthModel_;
    Model groundModel_;
    Shader lightShader_;

};

class Kdtree {
    public:
    Kdtree();
    ~Kdtree();
    void calcRobotNeighbours(std::map<int,std::shared_ptr<Robot>>& robots);

    typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> KDTree;
    Eigen::MatrixXd points_{};
    KDTree* kdtree_;
};

class Kdtree_formation {
    public:
    Kdtree_formation(Image& img);
    ~Kdtree_formation();
    void updateKdtree();
    void updateKdtree(Image& img, Eigen::Vector2d centre);
    void updateKdtree(Eigen::Vector2d new_centre);
    std::tuple<bool, std::vector<size_t>, std::vector<double>> search(Eigen::Vector2d  query_point, size_t num_results=1);
    std::pair<bool, std::vector<nanoflann::ResultItem<size_t, double>>> searchradius(Eigen::Vector2d  query_point, double search_radius=globals.WORLD_SZ/2.);

    Image formation_img_ = GenImageColor(125, 125, WHITE);
    typedef KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector2d>> KDTree_formation;
    std::vector<Eigen::Vector2d> points_{};
    KDTree_formation* kdtree_formation_;
    Eigen::Vector2d offset_{{0., 0.}};
    Eigen::Vector2d old_centre_{{0., 0.}};;
};

class Kdtree_obstacle {
    public:
    Kdtree_obstacle();
    ~Kdtree_obstacle();
    void updateKdtree();
    std::vector<nanoflann::ResultItem<size_t, double>> search(Eigen::Vector2d  query_point);
    Image obs_img_ = GenImageColor(125, 125, WHITE);
    void addObstacle(Eigen::Vector2d pos, Eigen::Vector2d vel, float rad);
    void updateObstaclePhysics();

    typedef KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector2d>> KDTree_obstacle;
    std::vector<Eigen::Vector2d> points_{};
    std::vector<Eigen::Vector2d> vels_{};
    std::vector<Rectangle> rects_{};
    std::vector<float> radii_{};
    KDTree_obstacle* kdtree_obstacle_;
    std::vector<Eigen::Vector4d> added_obstacle_{};
};

class KdtreeInfoVars {
    public:
    KdtreeInfoVars(std::vector<Eigen::Vector2d> points);
    ~KdtreeInfoVars();
    void updateKdtree();
    std::vector<nanoflann::ResultItem<size_t, double>> search(Eigen::Vector2d  query_point, double radius = 100.);

    typedef KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector2d>> KDtreeinfovars;
    std::vector<Eigen::Vector2d> points_{};
    KDtreeinfovars* tree;
};