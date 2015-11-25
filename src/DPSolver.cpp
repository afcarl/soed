
#include "DPSolver.h"

// solver initializes valueFunctions, making sure the last element is of type TerminalValueFunction
DPSolver::DPSolver(const int stages, std::shared_ptr<const Model> model)
{
  
}

// returns optimal control and value 
std::pair<double, double> GetOptimalControl(std::shared_ptr<const State> state, std::shared_ptr<const ValueFunction> nextValueFunction)
{

}

// solves Bellman's equation
void Solve(std::shared_ptr<const State> prior)
{
  // reset value functions
  valueFunctions.resize(numStages);
  for (int i = 0; i < numStages - 1; ++i)
    valueFunctions[i] = std::make_shared<ValueFunction>();
  valueFunctions[numStages - 1] = std::make_shared<TerminalValueFunction>(prior);
  
  ComputeTrainingPoints();
  TrainValueFunctions();
}

void ComputeTrainingPoints()
{
  
}

void TrainValueFunctions()
{
  
}
