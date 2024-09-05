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
    using Y = Eigen::Matrix<double, d, n>; 

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

        Y<d,n> y = Eigen::Matrix<double, d, N>::Zero();
        Y<d,n> dy = Eigen::Matrix<double, d, N>::Zero(); 
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

            Y<d,n> y = Eigen::Matrix<double, d, N>::Zero();
            Y<d,n> dy = Eigen::Matrix<double, d, N>::Zero(); 
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

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (n == Eigen::Dynamic || n > 1)
    void simpson_residual(OdeSystem<d,n>& system, 
            const double ts, 
            const Y<d,n>& y, Y<d,n>& f, J<d,n> *jf,
            Y<d,SubOneIfPositive<n>::r> &yc,
            Y<d,SubOneIfPositive<n>::r> &fc,
            J<d,SubOneIfPositive<n>::r> *jc,
            Y<d,SubOneIfPositive<n>::r> &res,
            Eigen::Matrix<double, MulIfPositive<SubOneIfPositive<n>::r,d>::r, MulIfPositive<n,d>::r> *jac
    ) {
        const int N = y.cols();
        constexpr int nl1 = SubOneIfPositive<n>::r;

        system.fun(y, f, jf);
        yc = 0.5 * (y.leftCols(N - 1) + y.rightCols(N - 1)) 
         + ts/8 * (f.leftCols(N - 1) - f.rightCols(N - 1));

        system.fun(yc, fc, jc);
        res = (f.leftCols(N - 1) + f.rightCols(N - 1) + 4 * fc);
        res = y.rightCols(N - 1) - y.leftCols(N - 1) - ts/6 * res ;

        if (jac == nullptr && jf != nullptr && jc != nullptr) return;

        (*jac).rightCols((N-1)*d) += Eigen::MatrixXd::Identity((N-1)*d, (N-1)*d) - ts/6 * (*jf).bottomRows((N-1)*d).rightCols((N-1)*d);
        (*jac).rightCols((N-1)*d) += -ts/3 * (*jc) + ts*ts/12 * (*jc) * (*jf).bottomRows((N-1)*d).rightCols((N-1)*d);

        (*jac).leftCols((N-1)*d) += -Eigen::MatrixXd::Identity((N-1)*d, (N-1)*d) - ts/6 * (*jf).topRows((N-1)*d).leftCols((N-1)*d);
        (*jac).leftCols((N-1)*d) += -ts/3 * (*jc) - ts*ts/12 * (*jc) * (*jf).topRows((N-1)*d).leftCols((N-1)*d);
    };
}
