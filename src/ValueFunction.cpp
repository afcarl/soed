#include <Eigen/Dense>
#include "ValueFunction.h"

using namespace std;
using namespace Eigen;

double ValueFunction::Evaluate(shared_ptr<const State> state)
{
  auto moments = state->GetMoments();
  double sum = 0;
  for (size_t i = 0; i < basisFunctions.size(); ++i) {
    sum += coefficients(i) * basisFunctions[i](moments.first, moments.second);
  }
  return sum; 
}

void ValueFunction::Train(const vector<shared_ptr<const State>> states, const VectorXd& costs)
{
  int m = static_cast<int>(states.size());
  int n = static_cast<int>(basisFunctions.size());
  MatrixXd X(m, n);
  for (int i = 0; i < m; ++i) {
    auto moments = states[i]->GetMoments();
    for (int j = 0; j < n; ++j) {
      X(i, j) = basisFunctions[j](moments.first, moments.second);
    }
  }
  coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * costs);
}

