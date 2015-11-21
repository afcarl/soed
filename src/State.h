#ifndef State_h
#define State_h

#include <tuple>
#include <vector>
#include <memory>

#include <Eigen/Core>

#include "Model.h"

class State
{

public:

  virtual double GetMean();
  
  virtual double GetVariance();
  
  virtual std::pair<double, double> GetMoments();
  
  virtual double GetKL(std::shared_ptr<State> other);
  
  virtual std::shared_ptr<State> GetNextState(std::shared_ptr<const Model>, const double control, const double disturbance);

};

#endif // ifndef State_h
