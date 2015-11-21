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
  
  State(const int numParticles) 
  
  inline std::pair<double, double> GetMoments()
  {
    double mean = std::accumulate(particles.begin(), particles.end(), 0.0) / numParticles;
    double variance = std::inner_product(particles.begin(), particles.end(), particles.begin(), 0.0) / numParticles - mean * mean;
    return std::make_pair(mean, variance);
  }
  
  inline double GetKL(std::shared_ptr<const State> other)
  {
    auto moments = GetMoments();
    double mu_1 = moments.first;
    double sigma_1 = sqrt(moments.second);
    
    auto otherMoments = other->GetMoments();
    double mu_2 = otherMoments.first;
    double sigma_2 = sqrt(otherMoments.second);
    
    return log(sigma_2) - log(sigma_1) + (sigma_1 * sigma_1 + (mu_1 - mu_2) * (mu_1 - mu_2)) / (2 * sigma_2 * sigma_2) - 0.5;
  }
  
  virtual std::shared_ptr<State> GetNextState(std::shared_ptr<const Model>, const double control, const double disturbance);

};

#endif // ifndef State_h
