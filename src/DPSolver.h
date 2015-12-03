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

  // returns optimal control and value as a std::pair
  std::pair<double, double> GetOptimalControl(std::shared_ptr<State> state, std::shared_ptr<ValueFunction> nextValueFunction);

  // returns optimal control and value as a std::pair, uses member variable valueFunctions indexed by stage
  std::pair<double, double> GetOptimalControl(std::shared_ptr<State> state, const int stage);

  std::pair<double, double> GetGreedyControl(std::shared_ptr<State> state);

  // solves Bellman's equation
  void Solve(std::shared_ptr<State> prior);

};

#endif // ifndef DPSolver_h
