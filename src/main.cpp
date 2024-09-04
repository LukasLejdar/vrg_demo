#include "fbvp/fbvp.hpp"
#include "fbvp/trajectory.hpp"
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iostream>

int main() {
    const int D = 4;
    const int N = 3;

    fbvp::T<> t = Eigen::MatrixXf::Constant(N, 1, 1);
    fbvp::Y<D> y = Eigen::MatrixXf::Constant(N, D, 1);
    fbvp::Y<D> dy = Eigen::MatrixXf::Constant(N, D, 0);
    fbvp::J<D> jac = Eigen::MatrixXf::Constant(N*D, N*D, 0);

    fbvp::OdeSystem<D> system;
    system.add_element(fbvp::velocity());
    system.add_element(fbvp::gravity());
    system.add_element(fbvp::air_drag(0.3, 1.293, 4.5e-5, 0.007));

    system.fun(t, y, dy, &jac);

    std::cout << y << std::endl;
    std::cout << dy << std::endl;
    std::cout << jac << std::endl;

}
