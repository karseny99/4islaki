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

template<typename T>
bool sign(T val) {
    return val > 0;
}

double origin_eq(double x) {
    return x * std::exp(x) + x*x - 1;
}

double function(double x) {
    // return std::log(1 - x*x) - std::log(x); // --> diverges
    // return sqrt(1 - x * std::exp(x)); // --> diverges
    // return (1 - x*x) * std::exp(-x); // --> diverges
  
    return (2*x - x*std::exp(x) + 1) / (x + 2);
}

std::function<double(double)> der(std::function<double(double)> f, double dx = 1e-10) {
    return [f, dx](double x) {
        return (f(x + dx) - f(x - dx)) / 2*dx;
    };
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

        auto df = solve::der(f);
        double q = std::max(fabs(df(current)), fabs(df(next))); 

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


double find_inital_point(
    std::function<double(double)> f, 
    std::function<double(double)> ddf, 
    double left, 
    double right, 
    double step = 1e-2
) {
    for(double guess = left; guess <= right; guess += step) {
        if(f(guess) * ddf(guess) > 0) {
            return guess;
        }
    }

    return (left + right) / 2;
}

root newton_method(
    std::function<double(double)> f,
    std::function<double(double)> df,
    std::function<double(double)> ddf,
    double left,
    double right,
    double eps = 1e-9, 
    size_t max_iter = 1000

) {

    // df = solve::der(f);
    // ddf = solve::der(df);

    if(f(left) * f(right) >= 0) 
        return {0, 0, ResultType::BAD_ROOT_INIT};
    
    double current = solve::find_inital_point(f, ddf, left, right);

    if(f(current) * ddf(current) <= 0) 
        return {current, 0, ResultType::BAD_ROOT_INIT};

    for (size_t iter(0); iter < max_iter; iter++) {
        double next = current - f(current) / df(current);

        if(solve::sign(df(current)) != solve::sign(df(next))) 
            return {next, iter + 1, ResultType::FUNCTION_DIVERGES};

        if(solve::sign(ddf(current)) != solve::sign(ddf(next)))
            return {next, iter + 1, ResultType::FUNCTION_DIVERGES};

        if(fabs(current - next) < eps) {
            return {next, iter + 1, ResultType::OK};
        }

        current = next;
    }
    
    return {current, max_iter, ResultType::MAX_ITER_EXCEEDED};
}

};


// int main() {
//     auto df = [](double x){ return std::exp(x)*(x + 1) + 2*x; };
//     auto ddf = [](double x){ return std::exp(x) * (x + 2) + 2; };

//     double eps = 1.0f;
//     for(size_t iter(0); iter < 40; ++iter, eps /= 2) {
//         auto simple_iteration = solve::simple_iteration_method(solve::function, 0.f, 1.f, eps);
//         auto newton_method = solve::newton_method(solve::origin_eq, df, ddf, 0.f, 1.f, eps);

//         std::cout << simple_iteration.iter_count << ' ' << newton_method.iter_count << ' ' << eps << '\n';
//     } 

// }

int main() {
    auto simple_iteration = solve::simple_iteration_method(solve::function, 0.f, 1.f, 1e-12);
    // std::cout << std::format("Return State is {}\nx={}\niteration count is {}\n", static_cast<int>(simple_iteration.state), simple_iteration.root, simple_iteration.iter_count);
    std::cout.precision(16);
    std::cout << "(Simple Iteration):\nReturn State is " << static_cast<int>(simple_iteration.state) << std::endl
              << "x=" << simple_iteration.root << std::endl 
              << "iteration count is " << simple_iteration.iter_count << std::endl;
    
    std::cout << std::endl;

    auto df = [](double x){ return std::exp(x)*(x + 1) + 2*x; };
    auto ddf = [](double x){ return std::exp(x) * (x + 2) + 2; };

    auto newton_method = solve::newton_method(solve::origin_eq, df, ddf, 0.f, 1.f, 1e-12);
    std::cout << "(Newton's method):\nReturn State is " << static_cast<int>(newton_method.state) << std::endl
              << "x=" << newton_method.root << std::endl 
              << "iteration count is " << newton_method.iter_count << std::endl;
}