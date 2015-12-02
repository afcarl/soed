#ifndef State_h
#define State_h

#include <cmath>
#include <tuple>
#include <memory>

#include <Eigen/Core>

#include "Model.h"
#include "RandomGenerator.h"

class State
{

public:

  Eigen::VectorXd particles;
  Eigen::VectorXd logWeights;

  inline void SetParticles(const Eigen::VectorXd& particles)   { this->particles  = particles; }
  inline void SetLogWeights(const Eigen::VectorXd& logWeights) { this->logWeights = logWeights; }

  State(const Eigen::VectorXd& particles, const Eigen::VectorXd& logWeights)
  {
    SetParticles(particles);
    SetLogWeights(logWeights);
  }

  inline std::shared_ptr<State> GetCopy()
  {
    return std::make_shared<State>(particles, logWeights);
  }

  inline std::pair<double, double> GetMoments()
  {
    auto weights = logWeights.array().exp();
    double sumWeights = weights.sum();
    double sumSquaredWeights = weights.square().sum();
    double mean = (weights * particles.array()).sum() / sumWeights;
    double variance = (weights * ((particles.array() - mean).square())).sum();
    variance = sumWeights / (sumWeights * sumWeights - sumSquaredWeights) * variance;
    return std::make_pair(mean, variance);
  }

  inline double GetKL(std::shared_ptr<State> other)
  {
    auto moments = GetMoments();
    double mu_1 = moments.first;
    double sigma_1 = sqrt(moments.second);

    auto otherMoments = other->GetMoments();
    double mu_2 = otherMoments.first;
    double sigma_2 = sqrt(otherMoments.second);

    return log(sigma_2) - log(sigma_1) + (sigma_1 * sigma_1 + (mu_1 - mu_2) * (mu_1 - mu_2)) / (2 * sigma_2 * sigma_2) - 0.5;
  }

  inline std::shared_ptr<State> GetNextState(std::shared_ptr<Model> model, const double control, const double disturbance)
  {
    auto newState = std::make_shared<State>(particles, logWeights);
    Eigen::VectorXd logLikelihoods(particles.size());
    for (int i = 0; i < particles.size(); ++i)
      logLikelihoods(i) = model->GetLogLikelihood(particles(i), control, disturbance);
    newState->logWeights += logLikelihoods;
    return newState;
  }

  inline double GetSample()
  {
    double sumWeights = logWeights.array().exp().sum();
    double sum = 0;
    double threshold = RandomGenerator::GetUniform();
    for (int i = 0; i < particles.size(); ++i) {
      double weight = exp(logWeights[i]);
      sum += weight;
      if (sum / sumWeights > threshold) {
        return particles[i];
      }
    }
    return particles[particles.size() - 1];
  }

};

#endif // ifndef State_h
