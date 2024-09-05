#include "fbvp/fbvp.hpp"
#include "fbvp/trajectory.hpp"
#include <cstdlib>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <system_error>

int main() {
    const int D = 4;
    const int N = 3;

    fbvp::OdeSystem<D> system;
    system.add_element(fbvp::velocity());
    system.add_element(fbvp::gravity());
    system.add_element(fbvp::air_drag(0.3, 1.293, 4.5e-5, 0.007));

    system.test_composition();

    fbvp::Y<D> y = Eigen::MatrixXd::Constant(D, N, 0);
    fbvp::Y<D> f = Eigen::MatrixXd::Constant(D, N, 0);
    fbvp::J<D> jf = Eigen::MatrixXd::Constant(N*D, N*D, 0);

    fbvp::Y<D> yc  = Eigen::MatrixXd::Constant(D, N-1, 0);
    fbvp::Y<D> fc  = Eigen::MatrixXd::Constant(D, N-1, 0);
    fbvp::J<D> jc  = Eigen::MatrixXd::Constant((N-1)*D, (N-1)*D, 0);

    fbvp::Y<D> res = Eigen::MatrixXd::Constant(D, N-1, 0);
    fbvp::J<D> jac = Eigen::MatrixXd::Constant((N-1)*D, N*D, 0);

    double theta = 0.1;
    double v0 = 350;
    y.row(0) = Eigen::ArrayXd::LinSpaced(N, 0, 1200);
    y.row(1) = Eigen::ArrayXd::LinSpaced(N, 0, 20);
    y.row(2).array() = v0 * cos(theta);
    y.row(3).array() = v0 * sin(theta);

    fbvp::simpson_residual<D>(system, 12.f/N, y, f, &jf, yc, fc, &jc, res, &jac);

    std::cout << jac << "\n\n";
}
