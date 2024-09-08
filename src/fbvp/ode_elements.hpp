#pragma once

#include "fbvp.hpp"
#include <eigen3/Eigen/src/Core/DenseBase.h>
#include <eigen3/Eigen/src/Core/Ref.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>

namespace fbvp {

    struct velocity: public JacFun {
        template<typename Y, typename F, typename J = MatNone>
        void fun(const Eigen::MatrixBase<Y>& y, Eigen::MatrixBase<F>& dy, Eigen::MatrixBase<J>* jac = nullptr) {
            constexpr int d = Y::RowsAtCompileTime;
            constexpr int N = Y::ColsAtCompileTime;
            static_assert(d == 4, "Supports only 2 dimensions");

            dy.row(0) += y.row(2);
            dy.row(1) += y.row(3);

            if (jac == nullptr) return;

            for (int i = 0; i < N; i++) {
                (*jac)(d*i + 0, d*i + 2) += 1;
                (*jac)(d*i + 1, d*i + 3) += 1;
            }
        } 
    };

    struct gravity : public JacFun {
        float g;
        gravity(float g) : g(g) {}

        template<typename Y, typename F, typename J = MatNone>
        void fun(const Eigen::MatrixBase<Y>& y, Eigen::MatrixBase<F>& dy, Eigen::MatrixBase<J>* jac = nullptr) {
            constexpr int d = Y::RowsAtCompileTime;
            constexpr int N = Y::ColsAtCompileTime;
            static_assert(d == 4, "Supports only 2 dimensions");

            dy.row(3).array() += g;
        }
    };

    struct air_drag: public JacFun {
        float drag_factor;

        air_drag(float drag_coef, float air_density, float area, float mass) 
            : drag_factor(drag_coef * air_density * area / (2*mass)) {}

        template<typename Y, typename F, typename J = MatNone>
        void fun(const Eigen::MatrixBase<Y>& y, Eigen::MatrixBase<F>& dy, Eigen::MatrixBase<J>* jac = nullptr) {
            constexpr int d = Y::RowsAtCompileTime;
            constexpr int N = Y::ColsAtCompileTime;
            static_assert(d == 4, "Supports only 2 dimensions");

            Eigen::Matrix<double, 1, Y::ColsAtCompileTime> v = 
                (y.row(2).array().square() + y.row(3).array().square()).array().sqrt();

            dy.row(2).array() -= drag_factor*v.array()*y.row(2).array();
            dy.row(3).array() -= drag_factor*v.array()*y.row(3).array();

            if (jac == nullptr) return;

            long rows = y.rows();
            for (int i = 0; i < y.cols(); i++) {
                double vi = v(i);
                if (abs(vi) == 0) vi = 1;

                (*jac)(rows*i + 2, rows*i + 2) -= drag_factor*(v(i) + y(2, i)*y(2, i)/vi );
                (*jac)(rows*i + 3, rows*i + 3) -= drag_factor*(v(i) + y(3, i)*y(3, i)/vi );
                (*jac)(rows*i + 2, rows*i + 3) -= drag_factor*y(2, i)*y(3, i)/vi;
                (*jac)(rows*i + 3, rows*i + 2) -= drag_factor*y(3, i)*y(2, i)/vi;
            }
        };
    };
}
