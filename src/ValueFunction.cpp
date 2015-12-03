#include <functional>
#include <iostream>
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
  auto moments = state->GetMoments();
  double sum = 0;
  for (size_t i = 0; i < basisFunctions.size(); ++i)
    sum += coefficients(i) * basisFunctions[i](moments.first, moments.second);
  return sum;
}

void ValueFunction::Train(const std::vector<std::shared_ptr<State>> states, const Eigen::VectorXd& values)
{
  trainingMeans = Eigen::VectorXd::Zero(states.size());
  trainingVariances = Eigen::VectorXd::Zero(states.size());
  trainingValues = Eigen::VectorXd::Zero(states.size());
  Eigen::MatrixXd X(states.size(), basisFunctions.size());
  for (size_t i = 0; i < states.size(); ++i) {
    auto moments = states[i]->GetMoments();
    trainingMeans(i) = moments.first;
    trainingVariances(i) = moments.second;
    trainingValues(i) = values(i);
    for (size_t j = 0; j < basisFunctions.size(); ++j)
      X(i, j) = basisFunctions[j](moments.first, moments.second);
  }
  coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * values);
  std::cout << coefficients.transpose() << std::endl;
}
