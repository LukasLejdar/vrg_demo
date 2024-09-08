#pragma once

#include "test.hpp"
#include <cassert>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/DenseBase.h>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorFixedSize.h>
#include <type_traits>


namespace fbvp {
    template <typename A0, typename... Args>
    void print(A0 a0, Args&&... args) {
        std::cout << a0;
        ((std::cout << ", " << args), ...);
        std::cout << "\n";
    }

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (n == Eigen::Dynamic || n > 0)
    using Y = Eigen::Matrix<double, d, n>; 

    template<unsigned int d, int n = Eigen::Dynamic>
    requires (n == Eigen::Dynamic || n > 0)
    using J = std::conditional_t<
        (n == Eigen::Dynamic),
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
        Eigen::Matrix<double, n*static_cast<int>(d), n*static_cast<int>(d)> 
    >;

    using MatNone = Eigen::Matrix<double, 0, 0>;

    template <typename T>
    struct TypeName {
        static void print() {
            std::cout << __PRETTY_FUNCTION__ << std::endl;
        }
    };

    struct JacFun {
        template <size_t d, size_t N, typename Y, typename F, typename J = MatNone>
        void fun(const Eigen::MatrixBase<Y>& y, Eigen::MatrixBase<F>& dy, Eigen::MatrixBase<J>* jac = nullptr);
    };

    template <typename... JacFuncs>
    class OdeSystem {
    public:
        OdeSystem(JacFuncs*... funcs) : funcs{funcs...} {}

        template <typename Y, typename F, typename J = MatNone>
        void fun(const Eigen::MatrixBase<Y>& y, Eigen::MatrixBase<F>& dy, Eigen::MatrixBase<J>* jac = nullptr) {
            constexpr int d = Y::RowsAtCompileTime;
            constexpr int N = Y::ColsAtCompileTime;
            static_assert(N > 0 && d > 0, "Dimensions must be known at compile time");

            _fun(y, dy, jac, std::make_index_sequence<sizeof...(JacFuncs)>{});
        }

        template<size_t d, size_t n>
        bool test_composition() {
            constexpr int N = (n < 1)? 3 : n;
            Y<d,n> y; Y<d,n> dy; J<d,n> jac;

            return test::jac_diferentiation(
                [this](const Y<d,n>& y, Y<d,n>& dy, J<d,n>* jac) { 
                    this->fun(y, dy, jac); }, y, dy, jac);
        }

    private:
        template <typename Y, typename F, typename J = MatNone, size_t... Ns>
        void _fun(const Eigen::MatrixBase<Y>& y, Eigen::MatrixBase<F>& dy, Eigen::MatrixBase<J>* jac, std::index_sequence<Ns...>) {
            (static_cast<JacFuncs*>(funcs[Ns])->fun(y, dy, jac), ...);
        }


        const std::array<JacFun*, sizeof...(JacFuncs)> funcs;
    };

    template <typename Y, typename F, typename... JacFuncs>
    void euler_step(OdeSystem<JacFuncs...> system, Eigen::MatrixBase<Y>& y, Eigen::MatrixBase<F>& dy, float ts) {
        assert(y.cols() == dy.cols() && y.rows() == dy.rows());

        dy.setZero();
        for (int i = 1; i < y.cols(); i++) {
            Eigen::Map<Eigen::Matrix<double, F::RowsAtCompileTime, 1>> dyi(dy.col(i-1).data());
            system.fun(y.col(i-1), dyi);
            y.col(i) = y.col(i-1) + ts*dyi;
        }
    }

    template <typename... JacFuncs, typename Y, typename F, typename Yc, typename Fc, typename Res, 
             typename Jf = MatNone, typename Jc = MatNone, typename Jac = MatNone>
    void simpson_residual(OdeSystem<JacFuncs...>& system, 
            const double ts, 
            Eigen::MatrixBase<Y> &y,
            Eigen::MatrixBase<F> &f,
            Eigen::MatrixBase<Yc> &yc,
            Eigen::MatrixBase<Fc> &fc,
            Eigen::MatrixBase<Res> &res,
            Eigen::MatrixBase<Jf> *jf = nullptr,
            Eigen::MatrixBase<Jc> *jc = nullptr,
            Eigen::MatrixBase<Jac> *jac = nullptr
    ) {
        f.setZero(); yc.setZero(); fc.setZero(); res.setZero(); 
        if (jac != nullptr && jf != nullptr && jc != nullptr) jf->setZero(); jc->setZero(); jac->setZero();

        //TODO: check f, yc, fc ... dimensions
        constexpr int d = Y::RowsAtCompileTime;
        constexpr int N = Y::ColsAtCompileTime;
        static_assert(N > 0 && d > 0, "Dimensions must be known at compile time");

        system.fun(y, f, jf);
        yc = 0.5 * (y.leftCols(N - 1) + y.rightCols(N - 1)) 
         + ts/8 * (f.leftCols(N - 1) - f.rightCols(N - 1));

        system.fun(yc, fc, jc);
        res = (f.leftCols(N - 1) + f.rightCols(N - 1) + 4.0 * fc);
        res = y.rightCols(N - 1) - y.leftCols(N - 1) - ts/6 * res ;

        if (jac == nullptr || jf == nullptr || jc == nullptr) return;

        (*jac).rightCols((N-1)*d) += Eigen::MatrixXd::Identity((N-1)*d, (N-1)*d) - ts/6 * (*jf).bottomRows((N-1)*d).rightCols((N-1)*d);
        (*jac).rightCols((N-1)*d) += -ts/3 * (*jc) + ts*ts/12 * (*jc) * (*jf).bottomRows((N-1)*d).rightCols((N-1)*d);

        (*jac).leftCols((N-1)*d) += -Eigen::MatrixXd::Identity((N-1)*d, (N-1)*d) - ts/6 * (*jf).topRows((N-1)*d).leftCols((N-1)*d);
        (*jac).leftCols((N-1)*d) += -ts/3 * (*jc) - ts*ts/12 * (*jc) * (*jf).topRows((N-1)*d).leftCols((N-1)*d);
    };

    template<typename _Y, typename... JacFuncs>
    void solve_ivp(OdeSystem<JacFuncs...> &system, Eigen::MatrixBase<_Y> &y, float ts) {
        constexpr int d = _Y::RowsAtCompileTime;
        constexpr int N = _Y::ColsAtCompileTime;
        static_assert(N > 0 && d > 0, "Dimensions must be known at compile time");

        Y<d,N> f; J<d,N> jf;
        Y<d,N-1> yc; Y<d,N-1> fc; J<d,N-1> jc;
        Y<d,N-1> res; Eigen::Matrix<double, d*(N-1), d*N> jac;

        // initialize result by euler method
        euler_step(system, y, f, ts); 
        
        // 2 Newton steps should be enough
        simpson_residual(system, ts, y, f, yc, fc, res, &jf, &jc, &jac); 
        Y<d*(N-1), 1> dy = jac.rightCols(d*(N-1)).colPivHouseholderQr().solve(res.reshaped(d*(N-1), 1));
        y.rightCols(N-1) -= dy.reshaped(d, N-1);

        simpson_residual(system, ts, y, f, yc, fc, res, &jf, &jc, &jac); 
        dy = jac.rightCols(d*(N-1)).colPivHouseholderQr().solve(res.reshaped(d*(N-1), 1));
        y.rightCols(N-1) -= dy.reshaped(d, N-1);

        simpson_residual(system, ts, y, f, yc, fc, res); 
        print("error2", res.array().square().sum());
    }
}
