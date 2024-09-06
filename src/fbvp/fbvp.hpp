#pragma once

#include "test.hpp"
#include <cassert>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorFixedSize.h>


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

    template <unsigned int d>
    struct JacFun {
        template<int n = Eigen::Dynamic>
        requires (is_dynamic_or_positive<n>())
        void fun(const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac);
    };


    template <unsigned int d, int... Ns>
    requires (sizeof...(Ns) > 0)
    class OdeSystem {
    public:
        template<typename T>
        void add_element(T& t) {
            (add_to_vtable<T, Ns>(t), ...);
            funcs.push_back(&t);
        }

        template<int n = Eigen::Dynamic>
        requires (is_dynamic_or_positive<n>())
        void fun(const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac) {
            assert(funcs.size() == vtable<n>().size());
            auto el = vtable<n>().begin();
            for (auto t = funcs.begin(); t != funcs.end(); el++, t++) {
                std::invoke(*el, *t, y, dy, jac);
            }
        }

        template<int n>
        requires (is_dynamic_or_positive<n>())
        bool test_composition() {
            constexpr int N = (n < 1)? 3 : n;

            Y<d,n> y = Y<d,n>::Zero(d, N);
            Y<d,n> dy = Y<d,n>::Zero(d, N); 
            J<d,n> jac = J<d,n>::Zero(d*N, d*N);

            return test::jac_diferentiation(
                [this](const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac) { 
                    this->fun(y, dy, jac); }, y, dy, jac);
        }

        bool test_composition() {
            return (test_composition<Ns>() && ...);
        }

    private:
        template <typename... Args>
        struct vtable_func {
            template <typename T>
            static void run(JacFun<d>* fun, Args... args) {
                const auto bound = [&](Args... args) {
                    static_cast<T*>(fun)->fun(args...);
                };

                std::invoke(bound, args...);
            }
        };

        template<typename T, int n>
        void add_to_vtable(JacFun<d>& fun) {
            vtable<n>().push_back(vtable_func<const Y<d,n>&, Y<d,n>&, J<d,n>*>::template run<T>);
        }
        
        template<int n>
        static auto& vtable() {
            static std::vector<void(*)(JacFun<d>*, const Y<d,n>&, Y<d,n>&, J<d,n>*)> vec;
            return vec;
        }

        std::vector<JacFun<d>*> funcs;
    };

    template<int n>
    struct SubOneIfPositive{
        static constexpr int r = (n > 0) ? n - 1 : n;
    };

    template<int n, unsigned int d>
    struct MulIfPositive{
        static constexpr int r = (n > 0) ? n*d : n;
    };

    template<unsigned int d, int n = Eigen::Dynamic, int nl1 = SubOneIfPositive<n>::r, int... Ns>
    requires (n == Eigen::Dynamic || n > 1)
    void simpson_residual(OdeSystem<d, Ns...>& system, 
            const double ts, 
            const Y<d,n>& y, Y<d,n>& f, J<d,n> *jf,
            Y<d,nl1> &yc, Y<d,nl1> &fc, J<d,nl1> *jc,
            Y<d,nl1> &res,
            Eigen::Matrix<double, MulIfPositive<nl1,d>::r, MulIfPositive<n,d>::r> *jac
    ) {
        const int N = y.cols();

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
