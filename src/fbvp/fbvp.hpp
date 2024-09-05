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
#include <functional>
#include <stdexcept>
#include <vector>
#include "test.hpp"

namespace fbvp {

    template<int n>
    constexpr bool is_dynamic_or_positive() {
        return (n > 0) || (n == Eigen::Dynamic);
    }

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_positive<n>())
    using Y = Eigen::Matrix<double, n, d>; 

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_positive<n>())
    using J = std::conditional_t<
        (n == -1),
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
        Eigen::Matrix<double, n*static_cast<int>(d), n*static_cast<int>(d)> 
    >;


    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_positive<n>())
    struct JacFunPair {
        std::function<void(const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac)> fun;
    };

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_positive<n>())
    bool test_jac_fun_pair( JacFunPair<d, n>& el) {
        const int N = (n == -1) ? 2 : n;

        Y<d,n> y = Eigen::Matrix<double, N, d>::Zero();
        Y<d,n> dy = Eigen::Matrix<double, N, d>::Zero(); 
        J<d,n> jac = Eigen::Matrix<double, N*d, N*d>::Zero();

        return test::jac_diferentiation(el.fun, y, dy, jac);
    }

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_positive<n>())
    class OdeSystem {
    public:
        OdeSystem() = default;
        OdeSystem(const std::vector<JacFunPair<d,n>>& elements) : elements(elements) {}

        void fun(const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac = nullptr) {
            assert(y.rows() == dy.rows());
            if (jac != nullptr) { assert(y.rows()*y.cols() == (*jac).rows() && (*jac).rows() == (*jac).cols()); }
            for(const auto& el : elements) {
                el.fun(y, dy, jac);
            }
        }

        template<typename U>
        requires std::is_same_v<std::remove_reference_t<U>, JacFunPair<d,n>>
        void add_element(U&& value) {
            bool valid = test_jac_fun_pair(value);
            if(!valid) throw std::runtime_error("Ode element is invalid\n");
            elements.emplace_back(std::forward<U>(value));
        }

        bool test_composition() {
            const int N = (n == -1) ? 2 : n;

            Y<d,n> y = Eigen::Matrix<double, N, d>::Zero();
            Y<d,n> dy = Eigen::Matrix<double, N, d>::Zero(); 
            J<d,n> jac = Eigen::Matrix<double, N*d, N*d>::Zero();

            return test::jac_diferentiation(
                [this](const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac) { this->fun(y, dy, jac); },
                y, dy, jac);
        }

    private:
        std::vector<JacFunPair<d,n>> elements;
    };

    template<int n>
    struct SubOneIfPositive{
        static constexpr int r = (n > 0) ? n - 1 : n;
    };

    template<int n, unsigned int d>
    struct MulIfPositive{
        static constexpr int r = (n > 0) ? n*d : n;
    };
}
