#ifndef DPSolver_h
#define DPSolver_h

#include <memory>
#include <tuple>
#include <vector>

#include "State.h"
#include "Model.h"

class DPSolver
{

private:

  int numStages;
  std::shared_ptr<Model> model;
  int numTrainingSamples;
  int numGridpoints;
  int numExpectation;
  std::vector<std::shared_ptr<ValueFunction>> valueFunctions;

public:

  inline void SetNumStages(const int numStages) { this->numStages = numStages; }
  inline void SetModel(std::shared_ptr<const Model> model) { this->model = model; }
  inline void SetNumTrainingSamples(const int numTrainingSamples) { this->numTrainingSamples = numTrainingSamples; }
  inline void SetNumExpectation(const int SetNumExpectation) { this->SetNumExpectation = SetNumExpectation; }
  inline void SetNumGridpoints(const int numGridpoints) { this->numGridpoints = numGridpoints; }
  
  // returns optimal control and value as a std::pair
  std::pair<double, double> GetOptimalControl(std::shared_ptr<const State> state, std::shared_ptr<const ValueFunction> nextValueFunction);

  // solves Bellman's equation
  void Solve(std::shared_ptr<const State> prior);
  
private:

  void ComputeTrainingPoints();
  void TrainValueFunctions();
  
};

#endif // ifndef DPSolver_h
