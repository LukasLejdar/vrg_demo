#pragma once

#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorFixedSize.h>
#include <vector>

namespace fbvp {

    template<int n>
    constexpr bool is_dynamic_or_unsigned() {
        return (n > 0) || (n == Eigen::Dynamic);
    }

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_unsigned<n>())
    using Y = Eigen::Matrix<float, n, d>; 

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_unsigned<n>())
    using J = std::conditional_t<
        (n == -1),
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>,
        Eigen::Matrix<float, n*static_cast<int>(d), n*static_cast<int>(d)> 
    >;


    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_unsigned<n>())
    struct JacFunPair {
        std::function<void(const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac)> fun;
    };

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (is_dynamic_or_unsigned<n>())
    class OdeSystem {
    public:
        OdeSystem() = default;
        OdeSystem(const std::vector<JacFunPair<d,n>>& elements) : elements(elements) {}

        void fun(const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac = nullptr) {
            for(const auto& el : elements) {
                el.fun(y, dy, jac);
            }
        }

        template<typename U>
        requires std::is_same_v<std::remove_reference_t<U>, JacFunPair<d,n>>
        void add_element(U&& value) {
            elements.emplace_back(std::forward<U>(value));
        }

    private:
        std::vector<JacFunPair<d,n>> elements;
    };

}
