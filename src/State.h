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
  double sumWeights;
  bool hasSumWeights;
  std::pair<double, double> moments;
  bool hasMoments;

  void SetParticles(const Eigen::VectorXd& particles)   { this->particles  = particles; }
  void SetLogWeights(const Eigen::VectorXd& logWeights) { this->logWeights = logWeights; }

  State(const Eigen::VectorXd& particles, const Eigen::VectorXd& logWeights);

  std::shared_ptr<State> GetCopy();

  std::pair<double, double> GetMoments();

  double GetKL(std::shared_ptr<State> other);

  std::shared_ptr<State> GetNextState(std::shared_ptr<Model> model, const double control, const double disturbance);

  double GetSample();

};

#endif // ifndef State_h
