/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <iostream>
#include <gbp/Simulator.h>
#include <gbp/Robot.h>
#include <nanoflann.h>

Simulator::Simulator(FormationName formation_name) : formation_name_(formation_name){
    // std::cout<<"Created Simulator"<<std::endl;
    SetTraceLogLevel(LOG_ERROR);  
    if (globals.CAPTURE){
        SetTargetFPS(60);
        InitWindow(globals.SCREEN_SZ, globals.SCREEN_SZ, globals.WINDOW_TITLE);
    }

    
    // Raylib camera uses Y axis as vertical. We use -Z axis vertical.
    // Therefor need to swap y and z values
    // Camera is define by a forward vector (target - position), as well as an up vector
    // Cycle through each set of positions, ups, targets by pressing 'k'
    // LARGE WORLD VIEW
    camera_positions_ = {(Vector3){ 0., 0.85f*globals.WORLD_SZ, 0.9f*globals.WORLD_SZ },
                        (Vector3){59.6969, 20.1162, 92.8933},
                        (Vector3){77.871, 20.1162, 15.8887}};
    camera_ups_ = {(Vector3){0.,0.,-1.}, (Vector3){-0.0460339, 0.389418, -0.91991}, (Vector3){-0.0460339, 0.389418, -0.91991}};
    camera_targets_ = {(Vector3){0.,0.,0.},
                         (Vector3){57, 0, 39},
                         (Vector3){75.1741, 0, -38.0046}};

    camera3d.position = camera_positions_[camera_idx_];
    camera3d.target = camera_targets_[camera_idx_];
    camera3d.up = camera_ups_[camera_idx_];             // Camera up vector
    camera3d.fovy = 60.0f;                              // Camera field-of-view Y
    camera3d.projection = CAMERA_PERSPECTIVE;           // Camera mode type

    kdtree_ = new Kdtree();

    // // For display only
    groundTruthImg = LoadImage(globals.BACKGROUND_FILE.c_str());
    if (globals.CAPTURE) texture_img = LoadTextureFromImage(groundTruthImg);
    if (globals.CAPTURE) graphics = new Graphics(this);
    ImageResize(&groundTruthImg, globals.WORLD_SZ, globals.WORLD_SZ);
    groundTruthField = LoadImageColors(groundTruthImg);
    int px_per_square = groundTruthImg.width/globals.gn; int start_x, start_y;
    
    for (int j=0; j<globals.gn; j++){
        start_y = j*px_per_square;
        for (int i=0; i<globals.gn; i++){
            start_x = i*px_per_square;
            //getminval from g[start_x to start_x+px_per_square, y..]
                float val = 1e3; int posx, posy;
                for (int col=start_x; col<start_x+px_per_square; col++){
                    for (int row=start_y; row<start_y+px_per_square; row++){
                        auto newval = ColorToHSV(groundTruthField[row * groundTruthImg.width + col]).z;
                        val = min(val, newval);
                    }
                }
                groundTruthMins.push_back(val);
        }
    }

 
};

Simulator::~Simulator(){
    delete kdtree_;
    if (globals.CAPTURE) delete graphics;
    for (int i = 0; i < robots_.size(); ++i) robots_.erase(i);
    if (globals.CAPTURE) UnloadTexture(texture_img);    
    if (globals.CAPTURE) CloseWindow();
};

int Simulator::addRobot(std::shared_ptr<Robot> robot){
    int new_rid = next_rid_++;
    robots_.insert_or_assign(new_rid, robot);
    return new_rid;
};

void Simulator::updateSetup(){
    /**** KEY PRESS HANDLER *****/
    int key;
    key = GetKeyPressed();
    switch (key){
        case KEY_K:
            globals.camera_flag = !globals.camera_flag;         // Toggles and begins transitions through Camera Views
            break;
        case KEY_P:
                globals.DRAW_PATH = !globals.DRAW_PATH;         // Display robot paths
            break;
        case KEY_I:
                globals.DRAW_INFO = !globals.DRAW_INFO;         // Display interrobot connections
            break;
        case KEY_G:
                globals.DRAW_GOAL = !globals.DRAW_GOAL;         // Display goal positions
            break;
        case KEY_SPACE:                                         // Pause simulation and restart simulation
            globals.SIM_MODE  = (globals.SIM_MODE==Timestep || globals.SIM_MODE==OneTimestep) ? SimNone : OneTimestep;
            break;
        case KEY_ESCAPE:                                        // Exit
            globals.RESET = globals.RUN = false;
            break;
        default:
            break;
    }

    /**** MOUSE PRESS HANDLER *****/
    ray_ = GetMouseRay(GetMousePosition(), camera3d);
    Vector3 mouse_gnd = Vector3Add(ray_.position, Vector3Scale(ray_.direction, -ray_.position.y/ray_.direction.y));
    Vector2 mouse_pos{mouse_gnd.x, mouse_gnd.z};
    // Do mouse logic here (eg. if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)){...} )

    /********* INITIAL ROBOT FORMATIONS **************/
    std::vector<std::vector<Eigen::VectorXd>> robots_to_create{};
    if (do_update_){
        do_update_ = false;
        switch (formation_name_){
        case FORMATION_SWARM_CORNER:
        {
            int square_sz = std::round(std::ceil(pow(globals.NUM_ROBOTS, 0.5)));
            int length = globals.ROBOT_RADIUS  * 4 * square_sz;
            int ox = -globals.WORLD_SZ/2;
            int oy = globals.WORLD_SZ/2;
            for (int n=0; n<globals.NUM_ROBOTS; n++){
                int start_x = ox + (n % square_sz) * (length/(float)square_sz);
                int start_y = oy - (n / square_sz) * (length/(float)square_sz);

                int end_x = start_x + 2.*(start_x - ox);
                int end_y = start_y + 2.*(start_x - ox);

                Eigen::VectorXd starting(globals.N_DOFS), ending(globals.N_DOFS);
                starting << start_x, start_y, 0., 0.;
                ending << end_x, end_y, 0., 0.;
                robots_to_create.push_back({starting, ending});
            }

            break;
        }
        case FORMATION_SWARM_RANDOM:
        {
            for (int n=0; n<globals.NUM_ROBOTS; n++){
                int start_x = random_number("uniform", -globals.WORLD_SZ/2.f, globals.WORLD_SZ/2.f);
                int start_y = random_number("uniform", -globals.WORLD_SZ/2.f, globals.WORLD_SZ/2.f);

                int end_x = start_x;
                int end_y = start_y;

                Eigen::VectorXd starting(globals.N_DOFS), ending(globals.N_DOFS);
                starting << start_x, start_y, 0., 0.;
                ending << end_x, end_y, 0., 0.;
                robots_to_create.push_back({starting, ending});
            }

            break;
        }
        case FORMATION_SWARM_LINE:
        {
            int ox = -globals.WORLD_SZ/2;
            int oy = 0.;
            for (int n=0; n<globals.NUM_ROBOTS; n++){
                int start_x = ox + (n+1) * (globals.WORLD_SZ/(float)(globals.NUM_ROBOTS+1));
                int start_y = oy;

                int end_x = start_x;
                int end_y = start_y;

                Eigen::VectorXd starting(globals.N_DOFS), ending(globals.N_DOFS);
                starting << start_x, start_y, 0., 0.;
                ending << end_x, end_y, 0., 0.;
                robots_to_create.push_back({starting, ending});
            }

            break;
        }
        case FORMATION_SWARM_CIRCLE:
        {
            for (int i=0; i<globals.NUM_ROBOTS; i++){
                float radius = globals.WORLD_SZ * 0.25;
                Eigen::VectorXd centre{{0., 0., 0.,0.}};

                Eigen::VectorXd pos_from_centre = Eigen::VectorXd{{radius * cos(2.*PI*i/(float)globals.NUM_ROBOTS)},
                                                                    {radius * sin(2.*PI*i/(float)globals.NUM_ROBOTS)},
                                                                    {0.},{0.}};
                Eigen::VectorXd starting = centre + pos_from_centre;
                Eigen::VectorXd ending = starting;
                robots_to_create.push_back({starting, ending});            
                }
            break;
        }
        default:
            break;
        }        

        // Create the actual robots and add to the simulation
        int idx=0;
        for (auto& wps : robots_to_create){
            float radius = globals.ROBOT_RADIUS;
            Color col = ColorFromHSV((idx++)*360./(float)robots_to_create.size(), 1., 0.75);
            auto robot = std::make_shared<Robot>(this, wps, radius, col);
            addRobot(robot);
        };
    };

    /******* UPDATE CAMERA AND CHECK FOR CAMERA TRANSITION *************/
    update_camera();
    if (globals.camera_flag){
        if (camera_clock_==(int)globals.p3){
            globals.camera_flag = false;
            camera_idx_ = (camera_idx_+1)%camera_positions_.size();
            camera_clock_ = 0;
        }
        camera3d.position = Vector3Lerp(camera_positions_[camera_idx_], camera_positions_[(camera_idx_+1)%camera_positions_.size()], (camera_clock_%(int)globals.p3)/globals.p3);
        camera3d.up = Vector3Lerp(camera_ups_[camera_idx_], camera_ups_[(camera_idx_+1)%camera_ups_.size()], (camera_clock_%(int)globals.p3)/globals.p3);
        camera3d.target = Vector3Lerp(camera_targets_[camera_idx_], camera_targets_[(camera_idx_+1)%camera_targets_.size()], (camera_clock_%(int)globals.p3)/globals.p3);
        camera_clock_++;
    }

};

/********** LOG METRICS FOR RESULTS ******************/
void Simulator::logMetrics(){
    float rms = 0;
    float coverage = 0;
    for (auto [rid, robot] : robots_){
        float sq_error_robot = 0;
        int count = 0;
        int coverage_count = 0;
        for (auto [vkey, var] : robot->fg[FGLayer::Information]->variables_){
            int idx = var->grid_idx.second*globals.gn + var->grid_idx.first;
            sq_error_robot += pow(var->mu_(2) - groundTruthMins[idx], 2.);
            count++;
            if (var->mu_(3)>0.) coverage_count++;
        }
        float rms_robot = pow(sq_error_robot / (float)(count), 0.5);
        float coverage_robot = (float)coverage_count/((float)globals.gn*globals.gn);
        robot->full_coverage_ = (coverage_count==(globals.gn*globals.gn));
        rms += rms_robot;
        coverage += coverage_robot;
        robot->positions_.push_back(robot->real_pose_({0,1}));
    }
    rms /= (float)globals.NUM_ROBOTS;
    coverage /= (float)globals.NUM_ROBOTS;
    rms_vector_.push_back(rms);
    coverage_vector_.push_back(coverage);

};

void Simulator::timestep(){

    if (globals.SIM_MODE==Timestep || globals.SIM_MODE==OneTimestep){
        logMetrics();

        // CALCULATE NEW NEIGHBOURS FOR ROBOTS
        kdtree_->calcRobotNeighbours(robots_);
        for (auto [r_id, robot] : robots_) {
            robot->updateInterrobotFactors();
        }
        // SAMPLE ENVIRONMENT
        if (clock_%globals.SAMPLE_TIMESTEP==0){
            for (auto [rid, robot] : robots_){
                Eigen::VectorXd pos = robot->fg[FGLayer::Planning]->getVar(0)->mu_({0,1});

                auto results = robot->kdtree_info_vars_->search(pos, pow(globals.MAPPING_RADIUS,2.));
                for (auto result : results){
                    // Get variable we are in
                    int idx = result.first;
                    float val = groundTruthMins[idx];
                    std::shared_ptr<Variable> variable = robot->fg[FGLayer::Information]->getVar(idx);
                    Eigen::VectorXd z{{variable->mu_(0), variable->mu_(1), robot->random_number("normal", val, globals.SIGMA_FACTOR_MAP), 1}};
                    Eigen::MatrixXd lam = Eigen::MatrixXd::Identity(4,4)*pow(globals.SIGMA_FACTOR_MAP, -2.);
                    lam(0,0) = lam(1,1) = lam(3,3) = pow(1e-5, -2.);
                    Eigen::VectorXd eta = lam * z;
                    variable->eta_prior_ += eta;
                    variable->lam_prior_ += lam;
                }
            }

        }


        // // Robot horizon pose update //
        int num_finished = 0;
        int num_covered = 0;
        for (auto [r_id, robot] : robots_) {
            robot->updateGoalFactors();         // 
            if (robot->found_) num_finished++;
            if (robot->full_coverage_) num_covered++;
        }
        source_found_ = (num_finished==globals.NUM_ROBOTS);
        full_coverage_ = (num_covered==globals.NUM_ROBOTS);
        if (globals.MODE==1){
            if (full_coverage_ && time_finished_==UINT32_MAX) time_finished_ = clock_;
        } else {
            if (source_found_ && time_finished_==UINT32_MAX) time_finished_ = clock_;
        }

        // Dropout        
        dropout_rids_.clear();
        std::vector<int> range{}; for (int r=0;r<globals.NUM_ROBOTS; r++) range.push_back(r);
        std::shuffle(range.begin(), range.end(), gen_uniform);
        for (int i=0; i<round(globals.DROPOUT*globals.NUM_ROBOTS); i++){
            dropout_rids_.push_back(range[i]);
        }
        for (auto [rid, robot] : robots_) {robot->inactive_info_layer_ = false;}
        for (auto f : dropout_rids_) {robots_[f]->inactive_info_layer_ = true;}

        iterate_gbp(globals.NUM_ITERS, FGLayer::Planning);
        if ((clock_%globals.SAMPLE_TIMESTEP==0)){
            iterate_gbp(globals.NUM_ITERS, FGLayer::Information);
            iterate_gbp(globals.NUM_ITERS, FGLayer::Goal);
        } else {
            iterate_gbp(globals.NUM_ITERS, FGLayer::Goal, true);
        }
        
        for (auto [r_id, robot] : robots_) {
            robot->updateHorizon();
            robot->updateCurrent();
        }
        clock_++;

    }
};

void Simulator::draw(){
    BeginDrawing();
        ClearBackground(BLACK);
        // ClearBackground(RAYWHITE);
        BeginMode3D(camera3d);
            // Draw Ground
            DrawModel(graphics->earthModel_, Vector3{0.f,globals.p2,0.f}, 1., WHITE);
            DrawModel(graphics->groundModel_, Vector3{0.f,globals.p1,0.f}, 1., WHITE);
            // Draw Robots
            for (auto [rid, robot] : robots_){
                robot->draw();
                for (auto rid : robot->connected_r_ids_){
                    if (robot->inactive_info_layer_ || robots_[rid]->inactive_info_layer_) continue;
                    DrawCylinderEx(Vector3{(float)(*robot)[0]->mu_(0), robot->height_3D_, (float)(*robot)[0]->mu_(1)},
                                    Vector3{(float)(*robots_[rid])[0]->mu_(0), robots_[rid]->height_3D_, (float)(*robots_[rid])[0]->mu_(1)}, 
                                    0.1, 0.1, 4, BLACK);
                }
            }

        EndMode3D();
    EndDrawing();    
    if (globals.TAKE_SCREENSHOTS){
        int n_zero = 6;
        std::string fname = globals.OUTPUT_SCREENSHOTS+std::string(n_zero - std::min(n_zero, (int)std::to_string(clock_).length()), '0') + std::to_string(clock_)+".png";
        TakeScreenshot(fname.data());
        print("Taking screenshot at ", fname.data());
    }
};

/****************************************************/
// GBP STUFF //
void Simulator::change_variable_prior(std::shared_ptr<Variable> p_var, const Eigen::VectorXd& new_mu){
    p_var->eta_prior_ = p_var->lam_prior_ * new_mu;
    p_var->mu_ = new_mu;
    for (auto [fkey, fac] : p_var->factors_){
        p_var->outbox_[fkey] = Message {p_var->eta_prior_, p_var->lam_prior_, p_var->mu_};
        p_var->inbox_[fkey].setZero();
    }
};

void Simulator::iterate_gbp(int n_iters, FGLayer layer, bool internal){
    for (int i=0; i<n_iters; i++){
#pragma omp parallel for
        // Iterate through robots
        for (int r_idx=0; r_idx<robots_.size(); r_idx++){
            internal = ((layer != FGLayer::Planning) && (std::find(dropout_rids_.begin(), dropout_rids_.end(), r_idx)!=dropout_rids_.end()));
            auto it_r = robots_.begin(); std::advance(it_r, r_idx);
            auto& [r_id, robot] = *it_r;
            if (layer==FGLayer::Information){
                robot->fg[layer]->factorIteration(internal, clock_);
            } else {
                robot->fg[layer]->factorIteration(internal);
            }
        }
    
#pragma omp parallel for
        // Iterate through robots
        for (int r_idx=0; r_idx<robots_.size(); r_idx++){
            internal = ((layer != FGLayer::Planning) && (std::find(dropout_rids_.begin(), dropout_rids_.end(), r_idx)!=dropout_rids_.end()));
            auto it_r = robots_.begin(); std::advance(it_r, r_idx);
            auto& [r_id, robot] = *it_r;
            if (layer==FGLayer::Information){
                robot->fg[layer]->variableIteration(internal, clock_);
            } else {
                robot->fg[layer]->variableIteration(internal);
            }
        }
    }
};


Graphics::Graphics(Simulator* sim) : sim_(sim){
    
    // Load basic lighting shader
    lightShader_ = LoadShader(PROJECT_DIR"assets/base_lighting.vs",
                            PROJECT_DIR"assets/lighting.fs");

    // Get some required shader locations
    lightShader_.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(lightShader_, "viewPos");
    
    // Ambient light level (some basic lighting)
    int ambientLoc = GetShaderLocation(lightShader_, "ambient");
    float temp[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    SetShaderValue(lightShader_, ambientLoc, temp, SHADER_UNIFORM_VEC4);

    // Assign our lighting shader to ground model
    groundModel_ = LoadModelFromMesh(GenMeshPlane(globals.WORLD_SZ, globals.WORLD_SZ, 10, 10));
    groundModel_.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = sim_->texture_img;
    // Assign our lighting shader to robot model
    robotModel_ = LoadModelFromMesh(GenMeshSphere(1., 50.0f, 50.0f));
    robotModel_.materials[0].shader = lightShader_;
    robotModel_.materials[0].maps[0].color = WHITE;
    // Assign our lighting shader to obstacle model
    earthModel_ = LoadModelFromMesh(GenMeshPlane(1.02*globals.WORLD_SZ, 1.02*globals.WORLD_SZ, 10, 10));
    earthModel_.materials[0].shader = lightShader_;
    earthModel_.materials[0].maps[0].color = BROWN;

    // Create lights
    Light lights[MAX_LIGHTS] = { 0 };
    Vector3 target = sim_->camera3d.target;
    Vector3 position = Vector3{target.x+10,target.y+20,target.z+10};
    lights[0] = CreateLight(LIGHT_POINT, position, target, LIGHTGRAY, lightShader_);                            
}

void Graphics::updateShader(){
    float cameraPos[3] = { sim_->camera3d.position.x, sim_->camera3d.position.y, sim_->camera3d.position.z };
    SetShaderValue(lightShader_, lightShader_.locs[SHADER_LOC_VECTOR_VIEW], cameraPos, SHADER_UNIFORM_VEC3);
}

Graphics::~Graphics(){};

Kdtree::Kdtree(){
    points_ = Eigen::MatrixXd(globals.NUM_ROBOTS, 2);
    kdtree_ = new KDTree(2, points_, 50);    

}
Kdtree::~Kdtree(){
    delete kdtree_;
}

void Kdtree::calcRobotNeighbours(std::map<int,std::shared_ptr<Robot>>& robots){
    for (auto [rid, robot] : robots){
        points_(rid, {0,1}) = (*robot)[0]->mu_({0,1});
    }
    kdtree_->index_->buildIndex();  

    for (auto [rid, robot] : robots){
        // Find nearest neighbors in radius
        robot->neighbours_.clear();
        Eigen::VectorXd query_pt = (*robots[rid])[0]->mu_({0,1});
        const float search_radius = pow(globals.COMMUNICATION_RADIUS,2.);
        std::vector<nanoflann::ResultItem<Eigen::Index, double>> matches;
        nanoflann::SearchParameters params; params.sorted = true;
        const size_t nMatches = kdtree_->index_->radiusSearch(&query_pt(0), search_radius, matches, params);
        for(size_t i = 0; i < nMatches; i++){
            if (matches[i].first!=rid && robot->neighbours_.size()<globals.ROBOT_MAX_NEIGHBOURS) robot->neighbours_.push_back(matches[i].first);
        }
    }
}

// Tree to store info variables to compute nearest ones
KdtreeInfoVars::KdtreeInfoVars(std::vector<Eigen::Vector2d> points){
    points_ = points;
    tree = new KDtreeinfovars(2, points_, 10);
    tree->index->buildIndex();
}
KdtreeInfoVars::~KdtreeInfoVars(){
    delete tree;
}

std::vector<nanoflann::ResultItem<size_t, double>> KdtreeInfoVars::search(Eigen::Vector2d query_pt, double radius){
    std::vector<nanoflann::ResultItem<size_t, double>> IndicesDists{};
    nanoflann::SearchParameters searchParams{};
    bool found = tree->index->radiusSearch(&query_pt[0], radius, IndicesDists, searchParams);
    return IndicesDists;
}

void Simulator::update_camera()
{
    float zoomscale = IsKeyDown(KEY_LEFT_SHIFT) ? 100. :10.;
    float zoom = -(float)GetMouseWheelMove() * zoomscale;
    CameraMoveToTarget(&camera3d, zoom);
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
    {
        Vector2 del = GetMouseDelta();
        // FOR UP {0,0,-1} and TOWARDS STRAIGHT DOWN
        if (IsKeyDown(KEY_LEFT_SHIFT)){
            CameraPitch(&camera3d, -del.y*0.05, true, true, true);                
            // Rotate up direction around forward axis
            camera3d.up = Vector3RotateByAxisAngle(camera3d.up, Vector3{0.,1.,0.}, -0.05*del.x);                
            Vector3 forward = Vector3Subtract(camera3d.target, camera3d.position);
            forward = Vector3RotateByAxisAngle(forward, Vector3{0.,1.,0.}, -0.05*del.x);
            camera3d.position = Vector3Subtract(camera3d.target, forward);
        } else if (IsKeyDown(KEY_LEFT_CONTROL)){
            float zoom = del.y*0.1;
            CameraMoveToTarget(&camera3d, zoom);
        } else {
            // Camera movement
            CameraMoveRight(&camera3d, -del.x, true);
            Vector3 D = GetCameraUp(&camera3d); D.y = 0.;
            D = Vector3Scale(Vector3Normalize(D), del.y);
            camera3d.position = Vector3Add(camera3d.position, D);
            camera3d.target = Vector3Add(camera3d.target, D);
        }
    }
}