
#include "../fbvp/fbvp.hpp"
#include "../fbvp/ode_elements.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

void find_elevation_angle(float xf, float yf, float zf, float vx, float vy, float vz,
        float v0, float mass, float drag_coef=0.3, float air_density=1.293, float area=4.5e-5) {

    const int N = 8;
    const int D = 6;

    double phi = atan(yf/xf);
    double theta = atan(zf/ sqrt(pow(xf, 2) + pow(yf, 2)) );
    double ts = xf / (v0 * cos(theta) * cos(phi) ) / N ;

    Eigen::Matrix<double, D, N> y;
    y.row(0) = Eigen::Matrix<double, 1, N>::LinSpaced(N, 0, xf);
    y.row(1) = Eigen::Matrix<double, 1, N>::LinSpaced(N, 0, yf);
    y.row(2) = Eigen::Matrix<double, 1, N>::LinSpaced(N, 0, zf);
    y.row(3).array() = v0 * cos(theta) * cos(phi);
    y.row(4).array() = v0 * cos(theta) * sin(phi);
    y.row(5).array() = v0 * sin(theta);

    fbvp::velocity vel{};
    fbvp::gravity grav{-9.8f};
    fbvp::air_drag drag{drag_coef, air_density, area, mass};
    fbvp::OdeSystem system{&vel, &grav, &drag};

    Eigen::Matrix<double, D-1, 1> bc_vars(theta, phi, y(3, N-1), y(4, N-1), y(5, N-1));

    fbvp::SetBC<D, N> set_bc = [v0, yf, xf, &zf, &vx, &vz, &vy](double ts, Eigen::Matrix<double, D-1, 1> bc_vars,
           Eigen::Map<Eigen::Matrix<double, D, 1>>& a, 
           Eigen::Map<Eigen::Matrix<double, D, 1>>& b) -> void {

        a(0) = 0;                     a(3) = v0 * cos(bc_vars[0]) * cos(bc_vars[1]);
        a(1) = 0;                     a(4) = v0 * cos(bc_vars[0]) * sin(bc_vars[1]);
        a(2) = 0;                     a(5) = v0 * sin(bc_vars[0]);

        // The the constants don't have to be so constant                                     
        b(0) = xf + vx*ts*(N-1);      b(3) = bc_vars(2); 
        b(1) = yf + vy*ts*(N-1);      b(4) = bc_vars(3);          
        b(2) = zf + vz*ts*(N-1);      b(5) = bc_vars(4);
    };

    fbvp::BCFunJac<D, N> bc_jac = [v0](double ts, double error, Eigen::Matrix<double, D-1, 1> bc_vars,
           const Eigen::Matrix<double, D*(N-1), D*N>& jac, 
           Eigen::Map<Eigen::Matrix<double, D*(N-1), D-1>>& bc_var_jac) -> void {

        std::cout << std::setprecision(10);
        std::cout << "theta " << bc_vars(0) << " psi " << bc_vars(1) << " error: " << error << "\n";

        bc_var_jac(Eigen::all, 1) = -jac(Eigen::all, 3) * v0 * cos(bc_vars[0]) * sin(bc_vars(1)) +jac(Eigen::all, 4) * v0 * cos(bc_vars[0]) * cos(bc_vars(1));
        bc_var_jac(Eigen::all, 0) = -jac(Eigen::all, 3) * v0 * sin(bc_vars[0]) * cos(bc_vars[1]) -jac(Eigen::all, 4) * v0 * sin(bc_vars[0]) * sin(bc_vars[1]);
        bc_var_jac(Eigen::all, 0) += jac(Eigen::all, 5) * v0 * cos(bc_vars[0]);
        bc_var_jac(Eigen::all, 2) = jac(Eigen::all, D*(N-1) + 3);
        bc_var_jac(Eigen::all, 3) = jac(Eigen::all, D*(N-1) + 4);
        bc_var_jac(Eigen::all, 4) = jac(Eigen::all, D*(N-1) + 5);

    };

    fbvp::solve_fbvp(system, y, &ts, bc_vars, set_bc, bc_jac, 50, 1e-6);

    std::cout << "\ny:\n";
    std::cout << y << "\n\n";

    std::cout << "Zásah v čase: " << ts*(N-1) << " s\n";

}

int main(int argc, char* argv[]) {
    std::vector<std::string> parameters = {"x1", "y1", "z1", "vx", "vy", "vz", "v0", "m"}; 
    std::array<float, 8> args = {1200, 0, 100, -100, 10, 5, 350,  0.007};
    size_t width = 7;

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

    std::cout << "Cíl se pohybuje konstantní rychlostí po dráze r(t) = (x1, y1, z1) + (vx, vy, vz)*t, zatímco střelec stojí v bodě (0, 0, 0)" << "\n\n";

    auto started = std::chrono::high_resolution_clock::now();
    find_elevation_angle(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);

    auto done = std::chrono::high_resolution_clock::now();
    std::cout << "\nexecution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << " ms \n";
}
