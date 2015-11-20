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

  virtual double GetMean() const;
  
  virtual double GetVariance() const;
  
  virtual std::pair<double, double> GetMoments() const;
  
  virtual double GetKL(std::shared_ptr<State> other) const;
  
  virtual std::shared_ptr<State> GetNextState(std::shared_ptr<const Model>, const double control, const double disturbance) const;

};

#endif // ifndef State_h
