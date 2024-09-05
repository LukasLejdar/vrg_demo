#include "fbvp/fbvp.hpp"
#include "fbvp/test.hpp"
#include "fbvp/trajectory.hpp"
#include <cstdlib>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iostream>

int main() {
    const int D = 4;
    const int N = 3;

    fbvp::OdeSystem<D> system;
    system.add_element(fbvp::velocity());
    system.add_element(fbvp::gravity());
    system.add_element(fbvp::air_drag(0.3, 1.293, 4.5e-5, 0.007));

    system.test_composition();

    fbvp::Y<D> y = Eigen::MatrixXd::Constant(N, D, 1);
    fbvp::Y<D> dy = Eigen::MatrixXd::Constant(N, D, 0);
    fbvp::J<D> jac = Eigen::MatrixXd::Constant(N*D, N*D, 0);

    system.fun(y, dy, &jac);

    std::cout << y << std::endl;
    std::cout << dy << std::endl;
    std::cout << jac << std::endl;

}
