#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono> // for benchmarking

// Box-Muller transform for standard normal
double randNormal() {
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main() {
    // Parameters
    double S0 = 100.0;   // initial stock price
    double K  = 100.0;   // strike
    double r  = 0.05;    // risk-free rate
    double sigma = 0.2;  // volatility
    double T = 1.0;      // time to maturity in years
    int N = 1000000;     // number of Monte Carlo simulations

    srand(time(NULL));   // seed RNG

    double sumPayoff = 0.0;

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Monte Carlo simulation
    for(int i = 0; i < N; i++) {
        double Z = randNormal();  // standard normal
        double ST = S0 * exp((r - 0.5*sigma*sigma)*T + sigma*sqrt(T)*Z); // risk-neutral stock
        double payoff = std::max(ST - K, 0.0); // call option payoff
        sumPayoff += payoff;
    }

    double price = exp(-r*T) * (sumPayoff / N);

    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output results
    std::cout << "European Call Option Price: " << price << std::endl;
    std::cout << "Simulation took: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}