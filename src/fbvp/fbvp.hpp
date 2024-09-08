#pragma once

#include "test.hpp"
#include <cassert>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/ArithmeticSequence.h>
#include <eigen3/Eigen/src/Core/DenseBase.h>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorFixedSize.h>
#include <iomanip>
#include <type_traits>


namespace fbvp {
    template <typename A0, typename... Args>
    void print(A0 a0, Args&&... args) {
        std::cout << a0;
        ((std::cout << " " << args), ...);
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
            static_assert(Y::ColsAtCompileTime > 0 && Y::RowsAtCompileTime > 0, "Dimensions must be known at compile time");
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
    void euler_method(OdeSystem<JacFuncs...> system, Eigen::MatrixBase<Y>& y, Eigen::MatrixBase<F>& dy, float ts) {
        assert(y.cols() == dy.cols() && y.rows() == dy.rows());

        dy.setZero();
        for (int i = 1; i < y.cols(); i++) {
            Eigen::Map<Eigen::Matrix<double, F::RowsAtCompileTime, 1>> dyi(dy.col(i-1).data());
            system.fun(y.col(i-1), dyi);
            y.col(i) = y.col(i-1) + ts*dyi;
        }
    }

    template <typename... JacFuncs, typename Y, typename F, typename Yc, typename Fc, typename Res, 
             typename Jf = MatNone, typename Jc = MatNone, typename Jac = MatNone, typename Z = MatNone>
    void simpson_residual(OdeSystem<JacFuncs...>& system, 
            const double ts, 
            const Eigen::MatrixBase<Y> &y,          // d x N
            Eigen::MatrixBase<F> &f,                // d x N
            Eigen::MatrixBase<Yc> &yc,              // d x (N-1)
            Eigen::MatrixBase<Fc> &fc,              // d x (N-1)
            Eigen::MatrixBase<Res> &res,            // d x (N-1)
            Eigen::MatrixBase<Jf> *jf = nullptr,    // d*N x d*N
            Eigen::MatrixBase<Jc> *jc = nullptr,    // d*(N-1) x d*(N-1)
            Eigen::MatrixBase<Jac> *jac = nullptr,   // d*(N-1) x d*N
            Eigen::MatrixBase<Res> *z = nullptr       // d x (N-1)
    ) {
        assert((jac != nullptr)? jc != nullptr && jf != nullptr : true);

        f.setZero(); yc.setZero(); fc.setZero(); res.setZero(); 
        if (jf != nullptr) jf->setZero();
        if (jc != nullptr) jc->setZero(); 
        if (jac != nullptr) jac->setZero();
        if (z != nullptr) z->setZero();


        //TODO: check f, yc, fc ... dimensions
        constexpr int d = Y::RowsAtCompileTime;
        constexpr int N = Y::ColsAtCompileTime;
        static_assert(N > 0 && d > 0, "Dimensions must be known at compile time");

        if (z == nullptr) z = &res;

        system.fun(y, f, jf);
        yc = 0.5 * (y.leftCols(N - 1) + y.rightCols(N - 1)) 
         + ts/8 * (f.leftCols(N - 1) - f.rightCols(N - 1));

        system.fun(yc, fc, jc);
        *z = (f.leftCols(N - 1) + f.rightCols(N - 1) + 4.0 * fc);
        res = y.rightCols(N - 1) - y.leftCols(N - 1) - ts/6 * (*z) ;

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

        // initialize result by euler
        euler_method(system, y, f, ts); 
        
        // 2 Newton steps should be enough
        simpson_residual(system, ts, y, f, yc, fc, res, &jf, &jc, &jac); 
        Y<d*(N-1), 1> dy = jac.rightCols(d*(N-1)).colPivHouseholderQr().solve(res.reshaped(d*(N-1), 1));
        y.rightCols(N-1) -= dy.reshaped(d, N-1);

        simpson_residual(system, ts, y, f, yc, fc, res, &jf, &jc, &jac); 
        dy = jac.rightCols(d*(N-1)).colPivHouseholderQr().solve(res.reshaped(d*(N-1), 1));
        y.rightCols(N-1) -= dy.reshaped(d, N-1);
    }

    template<int D, int N>
    using SetBC = std::function<void(float ts, Eigen::Matrix<double, D-1, 1> bc_vars,
            Eigen::Map<Eigen::Matrix<double, D, 1>>&, 
            Eigen::Map<Eigen::Matrix<double, D, 1>>&)>;

    template<int D, int N>
    using BCFunJac = std::function<void(float ts, Eigen::Matrix<double, D-1, 1> bc_vars, 
            Eigen::Matrix<double, D*(N-1), D*N>&, 
            Eigen::Map<Eigen::Matrix<double, D*(N-1), D-1>>&)>;

    // !!! won't work unless provided with a good initial guess for y, ts and bc_vars
    template<typename _Y, typename... JacFuncs, int d = _Y::RowsAtCompileTime, int N = _Y::ColsAtCompileTime>
    void solve_fbvp(OdeSystem<JacFuncs...> &system, Eigen::MatrixBase<_Y> &y, 
            float initial_ts, Eigen::Matrix<double, d-1, 1>& bc_vars, // need D variables for a uniqe solution. ts is one, the rest is bc_vars
            SetBC<d, N> set_bc, BCFunJac<d, N> bc_jacf
    ) {
        static_assert(N > 0 && d > 0, "Dimensions must be known at compile time");

        Y<d,N> f; J<d,N> jf;
        Y<d,N-1> yc; Y<d,N-1> fc; J<d,N-1> jc;
        Y<d,N-1> res; Eigen::Matrix<double, d*(N-1), d*N> jac;
        Y<d,N-1> z;
        Y<d*(N-1), 1> dy;

        double ts = initial_ts;

        Eigen::Map<Eigen::Matrix<double, d, 1>> a(y.col(0).data());
        Eigen::Map<Eigen::Matrix<double, d, 1>> b(y.col(N-1).data());
        Eigen::Map<Eigen::Matrix<double, d*(N-1), d-1>> bc_jac(jac(Eigen::all, Eigen::seq(d*(N-1) +1, d*(N-1) +d-1) ).data());

        for (int i = 0; i < 12; i++) {
            set_bc(ts, bc_vars, a, b);

            simpson_residual(system, ts, y, f, yc, fc, res, &jf, &jc, &jac, &z); 
            bc_jacf(ts, bc_vars, jac, bc_jac);
            jac(Eigen::all, d*(N-1)) = -z.reshaped(d*(N-1), 1)/6 -ts/12 * (jc * (f.leftCols(N - 1) - f.rightCols(N - 1)).reshaped(d*(N-1), 1));
            dy = jac.rightCols(d*(N-1)).colPivHouseholderQr().solve(res.reshaped(d*(N-1), 1));

            y.rightCols(N-1) -= dy.reshaped(d, N-1);
            bc_vars -= dy(Eigen::seq(d*(N-2) +1, d*(N-2) +d-1));
            ts -= dy(d*(N-2), 0);

            std::cout << std::setprecision(10);
            print("elevační úhel:", bc_vars(0), " error:", res.array().square().sum());

            float error = res.array().square().sum();
            if(error < 1e-20) break;
        }

        float error = res.array().square().sum();
        if(error > 1e-20) {
            std::cout << "\nPro zadané podmínky nelze cíl zasáhnout\n";
        }

    }
}
