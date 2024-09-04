#pragma once

#include "fbvp.hpp"

namespace fbvp {
    template<int n = Eigen::Dynamic>
    requires (is_dynamic_or_unsigned<n>())
    JacFunPair<4, n> velocity() {
        return JacFunPair<4, n>{
            [](const T<n>& t, const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
                dy.col(0) += y.col(2);
                dy.col(1) += y.col(3);

                if (jac == nullptr) return;

                long rows = y.rows();
                for (int i = 0; i < rows ; i++) {
                    (*jac)(rows*0 + i, rows*2 + i) += 1;
                    (*jac)(rows*1 + i, rows*3 + i) += 1;
                }
            },
        };
    };

    template<int n = Eigen::Dynamic>
    requires (is_dynamic_or_unsigned<n>())
    JacFunPair<4, n> gravity (float value = -9.8) {
        return JacFunPair<4, n>{
            [value](const T<n>& t, const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
                dy.col(3).array() += value;
            },
        };
    };


    template<int n = Eigen::Dynamic>
    requires (is_dynamic_or_unsigned<n>())
    JacFunPair<4, n> air_drag(float drag_coef, float air_density, float area, float mass) {
        float drag_factor = drag_coef * air_density * area / (2*mass);
        return JacFunPair<4, n>{
            [drag_factor](const T<n>& t, const Y<4, n>& y, Y<4, n>& dy, J<4, n>* jac) {
                T<n> v = (y.col(2).array().square() + y.col(3).array().square()).array().sqrt();
                dy.col(2).array() -= drag_factor*v.array()*y.col(2).array();
                dy.col(3).array() -= drag_factor*v.array()*y.col(3).array();

                if (jac == nullptr) return;

                long rows = y.rows();
                for (int i = 0; i < rows ; i++) {
                    float vi = v(i);
                    if (vi == 0) vi = 1;

                    (*jac)(rows*2 + i, rows*2 + i) -= drag_factor*(v(i) + y(i, 2)*y(i, 2)/vi );
                    (*jac)(rows*3 + i, rows*3 + i) -= drag_factor*(v(i) + y(i, 3)*y(i, 3)/vi );
                    (*jac)(rows*2 + i, rows*3 + i) -= drag_factor*y(i, 2)*y(i, 3)/vi;
                    (*jac)(rows*3 + i, rows*2 + i) -= drag_factor*y(i, 3)*y(i, 2)/vi;
                }
            },
        };
    };
}
