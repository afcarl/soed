#ifndef ValueFunction_h
#define ValueFunction_h

#include <Eigen/Core>

class ValueFunction
{

public:

  // Collection of basis functions that take the first two moments from State::GetMoments()
  static std::vector<std::function<double(const std::pair<double, double>)>> basisFunctions;

  Eigen::VectorXd coefficients;

  virtual double Evaluate(std::shared_ptr<const State> state);
  virtual double Train(const std::vector<std::shared_ptr<const State>> states, const std::vector<double> costs);

};

#endif // ifndef ValueFunction_h
