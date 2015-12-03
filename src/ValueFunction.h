#ifndef ValueFunction_h
#define ValueFunction_h

#include <vector>
#include <memory>

#include <Eigen/Core>

#include "State.h"

class ValueFunction
{

public:

  // Collection of basis functions that take the first two moments from State::GetMoments()
  static std::vector<std::function<double(double, double)>> basisFunctions;

  Eigen::VectorXd coefficients;
  Eigen::VectorXd trainingMeans;
  Eigen::VectorXd trainingVariances;
  Eigen::VectorXd trainingValues;

  inline Eigen::VectorXd GetCoefficients() { return coefficients; }
  virtual double Evaluate(std::shared_ptr<State> state);
  virtual void Train(const std::vector<std::shared_ptr<State>> states, const Eigen::VectorXd& values);

};

#endif // ifndef ValueFunction_h
