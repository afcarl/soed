#ifndef State_h
#define State_h

#include <tuple>
#include <vector>
#include <memory>
#include <numeric>

#include <Eigen/Core>

#include "Model.h"

class State
{

public:

  int numParticles;
  std::vector<double> particles;
  std::vector<double> logWeights;
  
  State(const int numParticles);
  std::pair<double, double> GetMoments();
  double GetKL(std::shared_ptr<const State> other);
  std::shared_ptr<State> GetNextState(std::shared_ptr<const Model>, const double control, const double disturbance);

};

#endif // ifndef State_h
