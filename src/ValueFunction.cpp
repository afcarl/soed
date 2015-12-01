#include <functional>
#include <Eigen/Dense>
#include "ValueFunction.h"

std::vector<std::function<double(double, double)>> ValueFunction::basisFunctions = {
  [](double mean, double variance) { return 1; },
  [](double mean, double variance) { return mean; },
  [](double mean, double variance) { return mean * mean; },
  [](double mean, double variance) { return log(sqrt(variance)); },
  [](double mean, double variance) { return variance; }
};

double ValueFunction::Evaluate(std::shared_ptr<State> state)
{
  auto moments = state->GetMoments();
  double sum = 0;
  for (size_t i = 0; i < basisFunctions.size(); ++i)
    sum += coefficients(i) * basisFunctions[i](moments.first, moments.second);
  return sum;
}

void ValueFunction::Train(const std::vector<std::shared_ptr<State>> states, const Eigen::VectorXd& costs)
{
  Eigen::MatrixXd X(states.size(), basisFunctions.size());
  for (size_t i = 0; i < states.size(); ++i) {
    auto moments = states[i]->GetMoments();
    for (size_t j = 0; j < basisFunctions.size(); ++j)
      X(i, j) = basisFunctions[j](moments.first, moments.second);
  }
  coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * costs);
}
