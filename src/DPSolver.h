#ifndef DPSolver_h
#define DPSolver_h

#include <memory>
#include <tuple>
#include <vector>

#include "State.h"
#include "Model.h"
#include "ValueFunction.h"
#include "TerminalValueFunction.h"

class DPSolver
{

public:

  int numStages;
  std::shared_ptr<Model> model;
  int numTrainingSamples;
  int numGridpoints;
  int numExpectation;
  std::vector<std::shared_ptr<ValueFunction>> valueFunctions;

  inline void SetNumStages(const int numStages) { this->numStages = numStages; }
  inline void SetModel(std::shared_ptr<Model> model) { this->model = model; }
  inline void SetNumTrainingSamples(const int numTrainingSamples) { this->numTrainingSamples = numTrainingSamples; }
  inline void SetNumExpectation(const int numExpectation) { this->numExpectation = numExpectation; }
  inline void SetNumGridpoints(const int numGridpoints) { this->numGridpoints = numGridpoints; }
  inline void SetValueFunctions(std::vector<std::shared_ptr<ValueFunction>> valueFunctions) {this->valueFunctions = valueFunctions; }

  // returns optimal control and value as a std::pair
  std::pair<double, double> GetOptimalControl(std::shared_ptr<State> state, std::shared_ptr<ValueFunction> nextValueFunction);

  // returns optimal control and value as a std::pair, uses member variable valueFunctions indexed by stage
  std::pair<double, double> GetOptimalControl(std::shared_ptr<State> state, const int stage);

  // solves Bellman's equation
  void Solve(std::shared_ptr<State> prior);

};

#endif // ifndef DPSolver_h
