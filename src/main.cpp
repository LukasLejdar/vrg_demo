#include "fbvp/fbvp.hpp"
#include "fbvp/ode_elements.hpp"
#include <cstdlib>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iostream>

int main() {
    const int D = 4; 
    const int N = 3;

    fbvp::velocity vel{};
    fbvp::gravity grav{-9.8f};
    fbvp::air_drag drag{0.3, 1.293, 4.5e-5, 0.007};

    fbvp::OdeSystem system{&vel, &grav, &drag};
    system.test_composition<D, N>();
    system.test_composition<D, N-1>();

    double theta = 0.1;
    double v0 = 350;

    fbvp::Y<D, N> y = Eigen::MatrixXd::Constant(D, N, 0);
    y.row(0) = Eigen::ArrayXd::LinSpaced(N, 0, 1200);
    y.row(1) = Eigen::ArrayXd::LinSpaced(N, 0, 20);
    y.row(2).array() = v0 * cos(theta);
    y.row(3).array() = v0 * sin(theta);

    fbvp::solve_ivp(system, y, 0.1);

    std::cout << y << "\n";
}
