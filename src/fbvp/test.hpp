#pragma once

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

namespace test {
    template<typename T>
    concept DereferencableToFloat = requires(T t) {
        { *t } -> std::same_as<float>;
    };

    // Define the concept is_iterable
    template <typename T>
    concept FloatIterable = requires(T v) {
        { v.data() } -> std::convertible_to<DereferencableToFloat>;
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

    template <typename F, FloatIterable X, FloatIterable Y>
    requires CallableWithXY<F, X, Y>
    bool aditivity( F fun, X& x, Y& y) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1000, 1000);

        for (auto it = x.data(); it != x.data() + x.size(); it++) *it = dis(gen);
        for (auto it = y.data(); it != y.data() + y.size(); it++) *it = 0;

        fun(x, y);
        for (auto it = y.data(); it != y.data() + y.size(); it++) *it *= -1;
        fun(x, y);
        for (auto it = y.data(); it != y.data() + y.size(); it++) {
            if (1e-6 > abs(*it)) continue;

            std::cout << "function does not fulfill the aditive requirement, y["
                << it - y.data() - y.size() << "] = " << *it << ", but should be 0\n";
            return false;
        }

        return true;
    }

    
    template <typename F, typename X, typename Y, typename J >
    requires CallableWithXYJ<F, X, Y, J>
    bool jac_diferentiation( F fun, X& x, Y& y, J& j) {
        assert(x.size()*y.size() == j.size());

        std::function<void(const X&, Y&)> yfun = [&fun](const X& x, Y& y) -> void {
            fun(x, y, nullptr);
        };

        std::function<void(const X&, J&)> jfun = [&fun, &y](const X& x, J& j) -> void {
            fun(x, y, &j);
        };

        if (!test::aditivity(yfun, x, y) || !test::aditivity(jfun, x, j)) {
            std::cout << "";
            return false;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1000, 1000);

        for (auto it = x.data(); it != x.data() + x.size(); it++) *it = dis(gen);
        for (auto it = y.data(); it != y.data() + y.size(); it++) *it = 0;
        for (auto it = j.data(); it != j.data() + j.size(); it++) *it = 0;
        
        fun(x, y, &j);

        float dt = 2e-5;
        auto jit = j.data();
        for (auto xit = x.data(); xit != x.data() + x.size(); ++xit) {

            for (auto yit = y.data(); yit != y.data() + y.size(); yit++) *yit = 0;       // y = 0
            *xit -= dt;
            fun(x, y, nullptr); // y = y1
            for (auto yit = y.data(); yit != y.data() + y.size(); yit++) *yit *= -1;     // y = -y1
            *xit += 2*dt;
            fun(x, y, nullptr);                                                          // y = y2 - y1
            *xit -= dt;
            for (auto yit = y.data(); yit != y.data() + y.size(); yit++) *yit /= (2*dt); // y = (y2 - y1) / (2*dt)

            for (auto yit = y.data(); yit != y.data() + y.size(); yit++, jit++) {
                if(1e-3 < abs(yit - jit)) continue;
                
                std::cerr << "Malformed fun jac[" 
                    << yit - y.data()  << ", " << xit - x.data()
                    << "] is " << *jit << " but should be " << *xit << "\n";
                return false;
            }
        }

        return true;
    }
}
