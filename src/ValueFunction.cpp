#include <Eigen/Dense>
#include "ValueFunction.h"

using namespace std;
using namespace Eigen;

double ValueFunction::Evaluate(shared_ptr<const State> state)
{
  auto moments = state->GetMoments();
  double sum = 0;
  for (size_t i = 0; i < basisFunctions.size(); ++i)
    sum += coefficients(i) * basisFunctions[i](moments.first, moments.second);
  return sum; 
}

void ValueFunction::Train(const vector<shared_ptr<const State>> states, const VectorXd& costs)
{
  MatrixXd X(states.size(), basisFunctions.size());
  for (size_t i = 0; i < states.size(); ++i) {
    auto moments = states[i]->GetMoments();
    for (size_t j = 0; j < basisFunctions.size(); ++j)
      X(i, j) = basisFunctions[j](moments.first, moments.second);
  }
  coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * costs);
}

