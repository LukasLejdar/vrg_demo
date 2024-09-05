#pragma once

#include "fbvp.hpp"

namespace fbvp {
    template<int n = Eigen::Dynamic>
    requires (is_dynamic_or_positive<n>())
    JacFunPair<4, n> velocity() {
        return JacFunPair<4, n>{
            [](const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
                dy.row(0) += y.row(2);
                dy.row(1) += y.row(3);

                if (jac == nullptr) return;

                int rows = y.rows();
                for (int i = 0; i < y.cols(); i++) {
                    (*jac)(rows*i + 0, rows*i + 2) += 1;
                    (*jac)(rows*i + 1, rows*i + 3) += 1;
                }
            },
        };
    };

    template<int n = Eigen::Dynamic>
    requires (is_dynamic_or_positive<n>())
    JacFunPair<4, n> gravity (double value = -9.8) {
        return JacFunPair<4, n>{
            [value](const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
                dy.row(3).array() += value;
            },
        };
    };


    template<int n = Eigen::Dynamic>
    requires (is_dynamic_or_positive<n>())
    JacFunPair<4, n> air_drag(double drag_coef, double air_density, double area, double mass) {
        double drag_factor = drag_coef * air_density * area / (2*mass);
        return JacFunPair<4, n>{
            [drag_factor](const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
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
            },
        };
    };
}
