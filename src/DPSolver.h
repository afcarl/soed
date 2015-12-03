#ifndef DPSolver_h
#define DPSolver_h

#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "State.h"
#include "Model.h"
#include "ValueFunction.h"
#include "TerminalValueFunction.h"

class DPSolver
{

public:

  int numStages;
  std::shared_ptr<Model> model;
  int numTrajectories;
  int numGridpoints;
  int numExpectation;
  std::vector<std::shared_ptr<ValueFunction>> valueFunctions;

  inline void SetNumStages(const int numStages) { this->numStages = numStages; }
  inline void SetModel(std::shared_ptr<Model> model) { this->model = model; }
  inline void SetNumTrajectories(const int numTrajectories) { this->numTrajectories = numTrajectories; }
  inline void SetNumExpectation(const int numExpectation) { this->numExpectation = numExpectation; }
  inline void SetNumGridpoints(const int numGridpoints) { this->numGridpoints = numGridpoints; }

  void SetValueFunctionCoefficients(const Eigen::MatrixXd& coefficients, std::shared_ptr<State> prior);

  Eigen::VectorXd GetOptimalValues(std::shared_ptr<State> state, std::shared_ptr<ValueFunction> nextValueFunction);
  Eigen::VectorXd GetOptimalValues(std::shared_ptr<State> state, const int stage);
  Eigen::VectorXd GetGreedyValues(std::shared_ptr<State> state);

  // solves Bellman's equation
  void Solve(std::shared_ptr<State> prior);

  Eigen::VectorXd GetControls();
  std::pair<double, double> GetControlPair(const Eigen::VectorXd& values);

};

#endif // ifndef DPSolver_h
