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
  
  // compute training points
  VectorXd priorSamples(numTrainingSamples);
  vector<vector<shared_ptr<State>>> trajectories; // for each prior sample, compute trajectory of states
  
  // for each trajectory (read: theta)
  //   for each stage
  //     get a disturbance using a random control
  //     get the next state using that random control and disturbance
  //     store the state at that stage
  for (int i = 0; i < numTrainingSamples; ++i) {
    trajectores[i].push_back(prior->GetCopy());
    for (int k = 1; k < numStages; ++k) {
      double control = RandomGenerator::GetUniform() * 2 - 1; // control between [-1, 1]
      double disturbance = model->GetDisturbance(priorSamples(i), control);
      trajectories[i].push_back(trajectories[i][k - 1]->GetNextState(model, control, disturbance));
    }
  }
  
  // for each state (in the trajectory at stage k) we need to find its value by calling GetOptimalControl
  // and we pass in that state and the next value function
  // and we get vector of states (for this stage)
  // and a vector of values, which we pass to Train
  for (int k = numStages - 2; k >= 0; --k) {
    vector<shared_ptr<State>> states;
    VectorXd values(numTrainingSamples);
    for (int i = 0; i < numTrainingSamples; ++i) {
      auto controlPair = GetOptimalControl(trajectories[i][k], valueFunctions[k + 1]);
      double control = controlPair.first;
      double value = controlPair.second;
      states.push_back(trajectories[i][k]);
      values(i) = value;
    }
    valueFunctions[k]->Train(states, values);
  }
  
}
