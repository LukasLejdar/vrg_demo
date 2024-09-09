
#pragma once

#include <catch2/catch_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include "../src/fbvp/finite_diff.hpp"
#include "../src/fbvp/ode_elements.hpp"
#include "../src/fbvp/fbvp.hpp"

bool almost_equal(double a, float b) {
 return std::fabs(a - b) <=  std::fmax(1e-4, 0.001 * std::fmax(std::fabs(a), std::fabs(b)));
}
    
template<int D1, int N1, int D2, int N2, typename F>
void test_jac_diff(F fun, int seed) {
    Eigen::Matrix<double, D1, N1> x;
    Eigen::Matrix<double, D2, N2> f;
    Eigen::Matrix<double, D2*N2, D1*N1> j_true;
    Eigen::Matrix<double, D2*N2, D1*N1> j_fun;

    f.setZero(); j_true.setZero(); j_fun.setZero();
    srand(seed);
    x.setRandom() * 1000;

    fun(x, f, &j_fun);
    fbvp::jac_finite_diff(fun, x, f, j_true);

    auto j_true_it = j_true.data();
    auto j_fun_it = j_fun.data();
    for (size_t i = 0; i < D1*N1; i++) {
        for (size_t j = 0; j < D2*N2; j++, j_true_it++, j_fun_it++) {

            INFO("Malformed fun jac[" 
               << j  << ", " << i 
               << "] is " << *j_fun_it << " but should be " << *j_true_it << "\n");
            INFO("\nJacFun jacobian calculation\n" << j_fun << "\n");
            INFO("\nfinite difference jacobian calculation\n" << j_true << "\n");
            REQUIRE(almost_equal(*j_true_it, *j_fun_it));
        }
    };
}

template<typename T, int D, int N>
void test_ode_element(fbvp::JacFun* jacFun, int seed) {
    auto fun = [&jacFun](const Eigen::Matrix<double, D, N> &y, Eigen::Matrix<double, D, N> &f, Eigen::Matrix<double, D*N, D*N> *j) -> void {
        static_cast<T*>(jacFun)->fun(y, f, j);
    };

    test_jac_diff<D,N,D,N>(fun, seed);
}

TEST_CASE("ode elements differentiation", "[differentiation]") {
    SECTION("velocity", "[velocity]") {
        fbvp::velocity velocity{};
        test_ode_element<fbvp::velocity, 4, 9>(&velocity, 1);
        test_ode_element<fbvp::velocity, 6, 4>(&velocity, 2);
    }

    SECTION("gravity", "[gravity]") {
        fbvp::gravity gravity{-9.8};
        test_ode_element<fbvp::gravity, 4, 7>(&gravity, 2);
        test_ode_element<fbvp::gravity, 6, 4>(&gravity, 6);
    }

    SECTION("air_drag", "[air_drag]") {
        fbvp::air_drag air_drag{0.3f, 1.293f, 4.5e-5f, 0.007f};
        test_ode_element<fbvp::air_drag, 4, 9>(&air_drag, 2);
        test_ode_element<fbvp::air_drag, 6, 5>(&air_drag, 6);
    }
}

TEST_CASE("simpson residual differentiation", "[differentiation]") {
}
