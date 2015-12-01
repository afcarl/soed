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
  VectorXd theta(numExpectation);
  VectorXd noise(numExpectation);
  for (int i = 0; i < numExpectation; ++i) {
    theta(i) = state->GetSample();
    noise(i) = model->GetNoiseSample();
  }
  for (int i = 0; i < numGridpoints; ++i) {
    double value = 0;
    for (int j = 0; j < numExpectation) {
      auto disturbance = model->GetDisturbance(theta(j), controls(i), noise(j));
      auto nextState = state->GetNextState(model, controls(i), disturbance);
      value += nextValueFunction->Evaluate(nextState);
    }
    values(i) = value / numExpectation;
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
