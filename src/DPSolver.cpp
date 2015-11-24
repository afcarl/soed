
#include "DPSolver.h"

// solver initializes valueFunctions, making sure the last element is of type TerminalValueFunction
DPSolver::DPSolver(const int stages, std::shared_ptr<const Model> model)
{
  
}

std::vector<std::shared_ptr<ValueFunction>> valueFunctions
{

}

// returns optimal control and value 
std::pair<double, double> GetOptimalControl(std::shared_ptr<const State> state, std::shared_ptr<const ValueFunction> nextValueFunction)
{

}

// solves Bellman's equation
void Solve(std::shared_ptr<const State> prior)
{
}

void ComputeTrainingPoints()
{
}

void TrainValueFunctions()
{
}
