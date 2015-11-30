#include <Eigen/Core>
#include "DPSolver.h"

using namespace std;
using namespace Eigen;

// returns optimal control and value 
pair<double, double> GetOptimalControl(shared_ptr<const State> state, shared_ptr<const ValueFunction> nextValueFunction)
{
  // enumerate all controls and values on a grid
  VectorXd controls = VectorXd::LinSpaced(-1, 1, numGridpoints); 
  VectorXd values = VectorXd::Zeros(numGridpoints); 
  for (int i = 0; i < numGridpoints; ++i) {
    // WHAT IS THETA??
    auto disturbance = model->GetDisturbance(theta, controls(i));
    auto nextState = state->GetNextState(model, controls(i), disturbance);
    values(i) = nextValueFunction->Evaluate(nextState);
  }
  // return optimal control and value among enumerated values 
  VectorXd::Index optimalIndex;
  double optimalValue = values.maxCoeff(&optimalIndex);
  double optimalControl = controls(optimalIndex);
  return make_pair<double, double>(optimalControl, optimalValue);
}

// solves Bellman's equation
void Solve(shared_ptr<const State> prior)
{
  // reset value functions
  valueFunctions.resize(numStages);
  for (int i = 0; i < numStages - 1; ++i)
    valueFunctions[i] = make_shared<ValueFunction>();
  valueFunctions[numStages - 1] = make_shared<TerminalValueFunction>(prior);
  
  ComputeTrainingPoints();
  TrainValueFunctions();
}

void ComputeTrainingPoints()
{
  
}

void TrainValueFunctions()
{
  
}
