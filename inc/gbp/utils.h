/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <functional>
#include <chrono>
#include <tuple>

#include <gbp/key.h>
#include <gbp/globals.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <raylib.h>
#include <rlgl.h>

#define LETTER_BOUNDRY_SIZE     0.25f
#define TEXT_MAX_LAYERS         32
#define LETTER_BOUNDRY_COLOR    VIOLET

extern Globals globals;

class Message {
    public:
    Eigen::VectorXd eta;
    Eigen::MatrixXd lambda;
    Eigen::VectorXd mu;

    Message(int n=globals.N_DOFS){
        eta = Eigen::VectorXd::Zero(n);
        lambda = Eigen::MatrixXd::Zero(n, n);
        mu = Eigen::VectorXd::Zero(n);
    }
    Message(Eigen::VectorXd eta_in, Eigen::MatrixXd lambda_in, Eigen::VectorXd mu_in = Eigen::VectorXd::Zero(globals.N_DOFS)){
        int n = eta_in.rows();
        eta = eta_in;
        lambda = lambda_in;
        mu = (mu_in.rows()==n) ? mu_in : Eigen::VectorXd::Zero(n);
    }
    
    Message& operator+=(const Message& msg_to_add) {eta += msg_to_add.eta; lambda += msg_to_add.lambda; return *this;};
    Message& operator-=(const Message& msg_to_add) {eta -= msg_to_add.eta; lambda -= msg_to_add.lambda; return *this;};
    const Message operator+(const Message& msg_to_add) const {Message ret_msg = *this; ret_msg += msg_to_add; return ret_msg;};
    const Message operator-(const Message& msg_to_sub) const {Message ret_msg = *this; ret_msg -= msg_to_sub; return ret_msg;};

    void setZero(){
        eta.setZero();
        lambda.setZero();
    }
    Message& setMu(Eigen::VectorXd mu_in) {this->mu = mu_in; return *this;};
};

using Mailbox = std::map<Key, Message>;


/********* Functions ******************************************************************/
void DrawTexturePro3D(Texture2D texture, Rectangle sourceRec, Rectangle destRec, Vector3 origin, float rotation, float posZ, Color tint);
Eigen::MatrixXd conv2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);

// Easy print statement
template <typename T> void print(const T& t) {
    std::cout << t << std::endl;
}
template <typename First, typename... Rest> void print(const First& first, const Rest&... rest) {
    std::cout << first << ", ";
    print(rest...); // recursive call using pack expansion syntax
}

void draw_info(uint32_t iter_cnt, uint32_t time_cnt, MODES_LIST mode, MOUSE_MODES_LIST mouse_mode, bool HELP=false);

template<typename Ta, typename Tb>
Tb min(Ta a, Tb b) {
    return ((Tb)a<b) ? (Tb)a:b;
}
template<typename Ta, typename Tb>
Tb max(Ta a, Tb b) {
    return ((Tb)a>b) ? (Tb)a:b;
}

std::vector<Eigen::VectorXd> linspace(Eigen::VectorXd low, Eigen::VectorXd high, int num_elems, bool upper_inclusive=false);

std::vector<int> get_variable_list(int H, int M);

// Draw circle within an image
// Copied and edited from raylib.h
void ImageDrawCircleFilled(Image *dst, int centerX, int centerY, int radius, Color color);

Color color_grad(Color start, Color end, float amount);
std::map<int, std::vector<std::tuple<int, float, float, float, float>>> read_record(std::string filename);
void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<int>>> dataset);

// TEMPLATE FOR TIMER FUNCTION
// Usage: 
// auto start = std::chrono::steady_clock::now();
// std::cout << "Elapsed(us): " << since(start).count() << std::endl;
template <
    class result_t   = std::chrono::microseconds,
    class clock_t    = std::chrono::high_resolution_clock,
    class duration_t = std::chrono::microseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

std::map<int, std::map<int, std::vector<float>>> read_csv(std::string filename);

// Computes squared Mahalanobis distance, given the observation z, mean mu and inverse covariance matrix Lam
float mahalanobis_distSquared(const Eigen::VectorXd& z, const Eigen::VectorXd& mu, const Eigen::MatrixXd& Lam);

void DrawText3D(Font font, const char *text, Vector3 position, float fontSize, float fontSpacing, float lineSpacing, bool backface, Color tint);
void DrawTextCodepoint3D(Font font, int codepoint, Vector3 position, float fontSize, bool backface, Color tint);
void DrawEllipse3D(Vector3 centerPos, Vector3 radius, int rings, int slices, Color color);
void DrawEllipse(Vector3 center, Vector2 radius, Vector3 rotationAxis, float rotationAngle, Color color);