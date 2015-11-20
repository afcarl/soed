#ifndef DPSolver_h
#define DPSolver_h

#include <memory>
#include <tuple>
#include <vector>

#include "State.h"
#include "Model.h"

class DPSolver
{

public:
  
  std::shared_ptr<Model> model;
  
  int stages;

  DPSolver(const int stages,std::shared_ptr<const Model> model) : stages(stages), model(model) { }

  std::vector<std::shared_ptr<ValueFunction>> valueFunctions;
  
  // returns optimal control and value 
  std::pair<double, double> GetOptimalControl(std::shared_ptr<const State> state, std::shared_ptr<const ValueFunction> nextValueFunction);

  // solves Bellman's equation
  void Solve(std::shared_ptr<const State> prior);
  
  void ComputeTrainingPoints();
  
  void TrainValueFunctions();

};

#endif // ifndef DPSolver_h
