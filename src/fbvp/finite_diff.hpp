
#pragma once

#include <catch2/catch_all.hpp>
#include <cassert>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorFixedSize.h>
#include <iostream>
#include <iterator>
#include <random>

namespace fbvp {
    template<typename T>
    concept DereferencableToDouble = requires(T t) {
        { *t } -> std::same_as<double>;
    };

    template <typename T>
    concept DoubleIterable = requires(T v) {
        { v.data() } -> std::convertible_to<DereferencableToDouble>;
        { v.data() } -> std::input_or_output_iterator;
        { v.size() } -> std::convertible_to<size_t>;
    };

    template<typename F, typename X, typename Y>
    concept CallableWithXY = requires(F f, const X& x, Y& y) {
        { f(x, y) } -> std::same_as<void>;
    };

    template<typename F, typename X, typename Y, typename J>
    concept CallableWithXYJ = requires(F f, const X& x, Y& y, J* j) {
        { f(x, y, j) } -> std::same_as<void>;
    };

    template <typename F, DoubleIterable X, DoubleIterable Y>
    requires CallableWithXY<F, X, Y>
    void aditivity( F fun, X& x, Y& y) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(-100, 100);

        for (auto it = x.data(); it != x.data() + x.size(); it++) *it = dis(gen);
        for (auto it = y.data(); it != y.data() + y.size(); it++) *it = 0;

        fun(x, y);
        for (auto it = y.data(); it != y.data() + y.size(); it++) *it *= -1;
        fun(x, y);
        for (auto yit = y.data(); yit != y.data() + y.size(); yit++) {
            int index = yit - y.data();
            if (1e-6 > abs(*yit)) continue;

            std::cout << "function does not fulfill the aditive requirement, y["
                << yit - y.data() << "] = " << *yit << ", but should be 0\n";
        }
    }

    bool almost_equal(double a, float b) {
     return std::fabs(a - b) <=  std::fmax(1e-4, 0.001 * std::fmax(std::fabs(a), std::fabs(b)));
    }
    
    template <typename F, DoubleIterable X, DoubleIterable Y, DoubleIterable J >
    requires CallableWithXYJ<F, X, Y, J>
    void jac_finite_diff( F fun, X& x, Y& y, J& j) {
        assert(x.size()*y.size() == j.size());

        for (auto it = y.data(); it != y.data() + y.size(); it++) *it = 0;

        auto jit = j.data();
        for (auto xit = x.data(); xit != x.data() + x.size(); ++xit) {

            double dt = std::max(abs(*xit * 1e-6), 1e-6);
            for (auto yit = y.data(); yit != y.data() + y.size(); yit++) *yit = 0;       // y = 0
            *xit -= dt;
            fun(x, y, nullptr);                                                          // y = y1
            for (auto yit = y.data(); yit != y.data() + y.size(); yit++) *yit *= -1;     // y = -y1
            *xit += 2*dt;
            fun(x, y, nullptr);                                                          // y = y2 - y1
            *xit -= dt;
            for (auto yit = y.data(); yit != y.data() + y.size(); yit++) *yit /= (2*dt); // y = (y2 - y1) / (2*dt)

            for (auto yit = y.data(); yit != y.data() + y.size(); yit++, jit++) *jit = (*yit == 0)? 0 : *yit;
        }
    }
}
