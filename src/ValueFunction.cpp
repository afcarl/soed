#include <functional>
#include <Eigen/Dense>
#include "ValueFunction.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
std::vector<std::function<double(double, double)>> ValueFunction::basisFunctions = {
  [](double mean, double variance) { return 1; },
  [](double mean, double variance) { return mean; },
  [](double mean, double variance) { return mean * mean; },
  [](double mean, double variance) { return log(sqrt(variance)); },
  [](double mean, double variance) { return variance; }
};
#pragma GCC diagnostic pop

double ValueFunction::Evaluate(std::shared_ptr<State> state)
{
  double sum = 0;
  for (size_t i = 0; i < basisFunctions.size(); ++i)
    sum += coefficients(i) * basisFunctions[i](state->moments.first, state->moments.second);
  return sum;
}

void ValueFunction::Train(const std::vector<std::shared_ptr<State>> states, const Eigen::VectorXd& costs)
{
  Eigen::MatrixXd X(states.size(), basisFunctions.size());
  for (size_t i = 0; i < states.size(); ++i) {
    for (size_t j = 0; j < basisFunctions.size(); ++j)
      X(i, j) = basisFunctions[j](states[i]->moments.first, states[i]->moments.second);
  }
  coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * costs);
}
