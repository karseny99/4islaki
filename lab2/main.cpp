#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iostream>
// #include <format>

// x*e^x + x^2 - 1 = 0
// --> (adding 2x for both sides)
// x*e^x + x^2 - 1 + 2x = 2x
// x*e^x + x(x + 2) - 1 = 2x


namespace solve {

double function(double x) {
    // return std::log(1 - x*x) - std::log(x); --> diverges
    // return sqrt(1 - x * std::exp(x)); --> diverges
    // return (1 - x*x) * std::exp(-x); --> diverges
    
    return (2*x - x*std::exp(x) + 1) / (x + 2);
}

double f_dx(std::function<double(double)> f, double x, double dx = 1e-4) {
    return (f(x + dx) - f(x)) / dx;
} 

enum class ResultType {
    OK,
    MAX_ITER_EXCEEDED,
    BAD_ROOT_INIT,
    FUNCTION_DIVERGES,
};

struct root {
    double root;
    size_t iter_count;
    ResultType state;
};

root simple_iteration_method(
    std::function<double(double)> f,
    double left,
    double right,
    double eps = 1e-9, 
    size_t max_iter = 1000
) {

    double current = (left + right) / 2;

    for(size_t iter(0); iter < max_iter; ++iter) {

        double next = f(current);
        double q = std::max(fabs(f_dx(f, current)), fabs(f_dx(f, next))); 

        if (q >= 1) {
            return {0, iter, ResultType::FUNCTION_DIVERGES};
        }
        
        // std::cout << "Iter " << iter << ": x = " << current 
        //          << ", next = " << next 
        //          << ", q = " << q << std::endl;

        if(fabs(current - next) < eps) {
            return {next, iter + 1, ResultType::OK};
        }

        current = next;
    }

    return {current, max_iter, ResultType::MAX_ITER_EXCEEDED};    
}

};


int main() {
    auto ans = solve::simple_iteration_method(solve::function, 0, 1.f);
    // std::cout << std::format("Return State is {}\nx={}\niteration count is {}\n", static_cast<int>(ans.state), ans.root, ans.iter_count);
    std::cout.precision(16);
    std::cout << "Return State is " << static_cast<int>(ans.state) << std::endl
              << "x=" << ans.root << std::endl 
              << "iteration count is " << ans.iter_count << std::endl;
}