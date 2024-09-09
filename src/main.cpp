#include "fbvp/fbvp.hpp"
#include "fbvp/ode_elements.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iomanip>
#include <iostream>
#include <vector>

template<size_t N> // number of collocation points
void find_elevation_angle(float xf, float yf, float v0, float mass, float drag_coef=0.3, float air_density=1.293, float area=4.5e-5) {
    const int D = 4;

    double theta = atan(yf/xf); // initial elevation angle guess
    double ts = xf / (v0 * cos(theta)) / N; // initial time step

    Eigen::Matrix<double, D, N> y; // x, y, vx, vy values at ts*n time samples up to n = N-1
    y.row(0) = Eigen::Matrix<double, 1, N>::LinSpaced(N, 0, xf); // x
    y.row(1) = Eigen::Matrix<double, 1, N>::LinSpaced(N, 0, yf); // y
    y.row(2).array() = v0 * cos(theta); // vx
    y.row(3).array() = v0 * sin(theta); // vy

    fbvp::velocity vel{};
    fbvp::gravity grav{-9.8f};
    fbvp::air_drag drag{drag_coef, air_density, area, mass};
    fbvp::OdeSystem system{&vel, &grav, &drag};

    Eigen::Matrix<double, D-1, 1> bc_vars(theta, y(2, N-1), y(3, N-1));

    fbvp::SetBC<D, N> set_bc = [v0, yf, xf](double ts, Eigen::Matrix<double, D-1, 1> bc_vars,
           Eigen::Map<Eigen::Matrix<double, D, 1>>& a, 
           Eigen::Map<Eigen::Matrix<double, D, 1>>& b) -> void {

        // the 4 constants        // the 3 bv variables + ts for a system of 4 differential equations
        a(0) = 0;                 a(2) = v0 * cos(bc_vars(0));
        a(1) = 0;                 a(3) = v0 * sin(bc_vars(0));
        b(0) = xf;                b(2) = bc_vars(1); 
        b(1) = yf;                b(3) = bc_vars(2);          
    };

    fbvp::BCFunJac<D, N> bc_jac = [v0](double ts, double error, Eigen::Matrix<double, D-1, 1> bc_vars,
           const Eigen::Matrix<double, D*(N-1), D*N>& jac, 
           Eigen::Map<Eigen::Matrix<double, D*(N-1), D-1>>& bc_var_jac) -> void {

        std::cout << std::setprecision(10);
        std::cout << "elevační úhel: " << bc_vars(0) << " error: " << error << "\n";

        bc_var_jac(Eigen::all, 0) = -jac(Eigen::all, 2) * sin(bc_vars(0)) * v0 + jac(Eigen::all, 3) * cos(bc_vars(0)) * v0;
        bc_var_jac(Eigen::all, 1) = jac(Eigen::all, D*(N-1) + 2);
        bc_var_jac(Eigen::all, 2) = jac(Eigen::all, D*(N-1) + 3);
    };

    fbvp::solve_fbvp(system, y, &ts, bc_vars, set_bc, bc_jac);

    std::cout << "\ny:\n";
    std::cout << y << "\n\n";

    std::cout << "Zásah v čase: " << ts*(N-1) << " s\n";
}

void find_elevation_angle(float x, float y, float z, float xf, float yf, float zf, 
        float v0, float mass, float drag_coef=0.3, float air_density=1.293, float area=4.5e-5) {

    float d = sqrt(pow(xf - x, 2) + pow(zf - z, 2));
    if (d == 0) {
        std::cout << "vzdálenost k cíli je 0\n";
        return;
    }
    find_elevation_angle<8>(d, yf - y, v0, mass, drag_coef, air_density, area);
}

int main(int argc, char* argv[]) {
    std::vector<std::string> parameters = {"x0", "y0", "z0", "x1", "y1", "z1", "v0", "m"}; 
    std::array<float, 8> args = {0, 0, 0, 1200, 20, 0, 350, 0.007};
    size_t width = 5;

    for (int i = 1; i < 9 && i < argc; i++) {
        try {
            args[i-1] = std::stof(argv[i]);
            width = std::max(width, std::strlen(argv[i])+2);
        }  catch (const std::invalid_argument& e) {
            std::cerr << "Neplatný argument: " << e.what() << "\n";
        }
    }

    std::cout << std::left << std::setw(4) << parameters[0] << std::setw(width) << args[0];
    std::cout << std::setw(4) << parameters[3] << std::setw(width) << args[3] << "\n";

    std::cout << std::left << std::setw(4) << parameters[1] << std::setw(width) << args[1];
    std::cout << std::setw(4) << parameters[4] << std::setw(width) << args[4] << "\n";

    std::cout << std::left << std::setw(4) << parameters[2] << std::setw(width) << args[2];
    std::cout << std::setw(4) << parameters[5] << std::setw(width) << args[5] << "\n";
    
    std::cout << std::left << std::setw(4) << parameters[6] << std::setw(width) << args[6];
    std::cout << std::setw(4) << parameters[7] << std::setw(width) << args[7] << "\n";
    std::cout << "\n";

    auto started = std::chrono::high_resolution_clock::now();
    find_elevation_angle(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);

    auto done = std::chrono::high_resolution_clock::now();
    std::cout << "\nexecution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << " ms \n";
}
