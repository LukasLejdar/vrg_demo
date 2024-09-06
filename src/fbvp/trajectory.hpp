#pragma once

#include "fbvp.hpp"

namespace fbvp {

    struct velocity: public JacFun<4> {

        template<int n>
        void fun(const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
            dy.row(0) += y.row(2);
            dy.row(1) += y.row(3);

            if (jac == nullptr) return;

            int rows = y.rows();
            for (int i = 0; i < y.cols(); i++) {
                (*jac)(rows*i + 0, rows*i + 2) += 1;
                (*jac)(rows*i + 1, rows*i + 3) += 1;
            }
        }
    };

    struct gravity : public JacFun<4> {
        float g;
        gravity(float g) : g(g) {}

        template<int n>
        void fun(const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
            dy.row(3).array() += g;
        }
    };

    struct air_drag: public JacFun<4> {
        double drag_factor;

        air_drag(double drag_coef, double air_density, double area, double mass) 
            : drag_factor(drag_coef * air_density * area / (2*mass)) {}

        template<int n>
        void fun(const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
                Eigen::Matrix<double, 1, n> v = (y.row(2).array().square() + y.row(3).array().square()).array().sqrt();
                dy.row(2).array() -= drag_factor*v.array()*y.row(2).array();
                dy.row(3).array() -= drag_factor*v.array()*y.row(3).array();

                if (jac == nullptr) return;

                long rows = y.rows();
                for (int i = 0; i < y.cols(); i++) {
                    double vi = v(i);
                    if (vi == 0) vi = 1;

                    (*jac)(rows*i + 2, rows*i + 2) -= drag_factor*(v(i) + y(2, i)*y(2, i)/vi );
                    (*jac)(rows*i + 3, rows*i + 3) -= drag_factor*(v(i) + y(3, i)*y(3, i)/vi );
                    (*jac)(rows*i + 2, rows*i + 3) -= drag_factor*y(2, i)*y(3, i)/vi;
                    (*jac)(rows*i + 3, rows*i + 2) -= drag_factor*y(3, i)*y(2, i)/vi;
                }
        };
    };
}
