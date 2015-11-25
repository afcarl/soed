#ifndef State_h
#define State_h

#include <tuple>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>

#include "Model.h"

class State
{
private:

  std::vector<double> particles;
  std::vector<double> logWeights;
  
public:

  inline void SetParticles(const std::vector<double>& particles) { this->particles = particles; }
  inline void SetLogWeights(const std::vector<double>& logWeights) { this->logWeights = logWeights; }
  
  inline void AddParticle(const double particle, const double logWeight = 0) 
  { 
    particles.push_back(particle); 
    logWeights.push_back(logWeight);
  }

  inline void NormalizeWeights() const
  {
    double maxCoeff = *std::max_element(logWeights.begin(), logWeights.end());
    double sumExp = 0;
    for (size_t i = 0; i < logWeights.size(); ++i)
      sumExp += exp(logWeights - maxCoeff);
    double logSumExp = maxCoeff + log(sumExp);
    
    for (size_t i = 0; i < logWeights.size(); ++i)
      logWeights[i] -= logSumExp;
  }

  inline std::pair<double, double> GetMoments() const
  {
    double mean = 0;
    double sumWeights = 0;
    for (size_t i = 0; i < particles.size(); ++i) {
      double weight = exp(logWeights[i]);
      sumWeights += weight;
      mean += weight * particles[i];
    }
    mean = mean / sumWeights;

    double variance = 0;
    double sumSquaredWeights = 0;
    for (size_t i = 0; i < particles.size(); ++i) {
      double weight = exp(logWeights[i]);
      sumSquaredWeights += weight * weight;
      variance += weight * (particles[i] - mean) * (particles[i] - mean);
    }
    variance = sumWeights / (sumWeights * sumWeights - sumSquaredWeights) * variance;
    
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
  
  inline std::shared_ptr<State> GetNextState(std::shared_ptr<const Model>, const double control, const double disturbance) const
  {
    auto std::make_shared<State> newState();
    for (size_t i = 0; i < particles.size(); ++i)
      newState->AddParticle(particles[i], logWeights[i] + model->GetLogLikelihood(particles[i], control, disturbance));
    return newState;
  }

};

#endif // ifndef State_h
