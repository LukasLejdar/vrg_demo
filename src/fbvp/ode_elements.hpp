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
            static_assert(d % 2 == 0, "The number of ODEs must be even so that every spatial dimension can have its derivative");

            constexpr int dim = d / 2;

            for (int i = 0; i < dim; i++) {
                dy.row(i) += y.row(dim+i);
            }

            if (jac == nullptr) return;

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < dim; j++) {
                    (*jac)(d*i + j, d*i + dim +j) += 1;
                }
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
            static_assert(d % 2 == 0, "The number of ODEs must be even so that every spatial dimension can have its derivative");

            dy.row(d-1).array() += g;
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
            static_assert(d % 2 == 0, "The number of ODEs must be even so that every spatial dimension can have its derivative");

            constexpr int dim = d / 2;
            
            Eigen::Matrix<double, 1, Y::ColsAtCompileTime> v;
            v.setZero();
            for (int i = 0; i < dim; i++) v += y.row(dim+i).array().square().matrix();
            v = v.array().sqrt();

            for (int i = 0; i < dim; i++) {
                dy.row(dim+i).array() -= drag_factor*v.array()*y.row(dim+i).array();
            }

            if (jac == nullptr) return;

            long rows = y.rows();
            for (int i = 0; i < y.cols(); i++) {
                double vi = v(i);
                if (abs(vi) == 0) vi = 1;

                for (int j = dim; j < d; j++) { 
                    (*jac)(rows*i + j, rows*i + j) -= drag_factor*(v(i));
                    for (int k = dim; k < d; k++) { 
                        (*jac)(rows*i + j, rows*i + k) -= drag_factor*(y(j, i)*y(k, i)/vi );
                    }
                }
            }
        };
    };
}
