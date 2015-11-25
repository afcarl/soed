#ifndef ValueFunction_h
#define ValueFunction_h

#include <vector>
#include <memory>

#include <Eigen/Core>

#include "State.h"

class ValueFunction
{
  
private:

  // Collection of basis functions that take the first two moments from State::GetMoments()
  static std::vector<std::function<double(double, double)>> basisFunctions;

  Eigen::VectorXd coefficients;
  
public:

  inline Eigen::VectorXd GetCoefficients() { return coefficients; }
  virtual double Evaluate(std::shared_ptr<const State> state);
  virtual void Train(const std::vector<std::shared_ptr<const State>> states, const Eigen::VectorXd& costs);

};

#endif // ifndef ValueFunction_h
