#include <cstdio>
#include <Eigen/Core>
#include "DPSolver.h"

using namespace std;
using namespace Eigen;

// returns optimal control and value
pair<double, double> DPSolver::GetOptimalControl(shared_ptr<State> state, shared_ptr<ValueFunction> nextValueFunction)
{
  // enumerate all controls and values on a grid
  //
  VectorXd controls = VectorXd::LinSpaced(numGridpoints, -1, 1);
  VectorXd values = VectorXd::Zero(numGridpoints);
  VectorXd theta(numExpectation);
  VectorXd noise(numExpectation);
  for (int i = 0; i < numExpectation; ++i) {
    theta(i) = state->GetSample();
    noise(i) = model->GetNoiseSample();
  }
  for (int i = 0; i < numGridpoints; ++i) {
    double value = 0;
    for (int j = 0; j < numExpectation; ++j) {
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
  return make_pair(optimalControl, optimalValue);
}

pair<double, double> DPSolver::GetOptimalControl(shared_ptr<State> state, const int stage)
{
  return GetOptimalControl(state, valueFunctions[stage + 1]);
}

// solves Bellman's equation
void DPSolver::Solve(shared_ptr<State> prior)
{
  // reset value functions
  valueFunctions.resize(numStages);
  for (int i = 0; i < numStages - 1; ++i)
    valueFunctions[i] = make_shared<ValueFunction>();
  valueFunctions[numStages - 1] = make_shared<TerminalValueFunction>(prior);

  printf("Computing training points\n");

  // compute training points
  VectorXd priorSamples(numTrainingSamples);
  vector<vector<shared_ptr<State>>> trajectories(numTrainingSamples); // for each prior sample, compute trajectory of states

  // for each trajectory (read: theta)
  //   for each stage
  //     get a disturbance using a random control
  //     get the next state using that random control and disturbance
  //     store the state at that stage
  for (int i = 0; i < numTrainingSamples; ++i) {
    printf("%d/%d\r", i+1, numTrainingSamples); fflush(stdout);
    trajectories[i].push_back(prior->GetCopy());
    for (int k = 1; k < numStages; ++k) {
      double control = RandomGenerator::GetUniform() * 2 - 1; // control between [-1, 1]
      double disturbance = model->GetDisturbance(priorSamples(i), control);
      trajectories[i].push_back(trajectories[i][k - 1]->GetNextState(model, control, disturbance));
    }
  }
  printf("\nDone\n");

  printf("Training value functions\n");
  // for each state (in the trajectory at stage k) we need to find its value by calling GetOptimalControl
  // and we pass in that state and the next value function
  // and we get vector of states (for this stage)
  // and a vector of values, which we pass to Train
  for (int k = numStages - 2; k >= 0; --k) {
    printf("Computing values for stage k = %d\n", k);
    vector<shared_ptr<State>> states(numTrainingSamples);
    VectorXd values(numTrainingSamples);
    #pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < numTrainingSamples; ++i) {
      printf("%d/%d\r", i+1, numTrainingSamples); fflush(stdout);
      auto controlPair = GetOptimalControl(trajectories[i][k], valueFunctions[k + 1]);
      double control = controlPair.first;
      double value = controlPair.second;
      states[i] = trajectories[i][k];
      values(i) = value;
    }
    printf("Computing linear least squares\n");
    valueFunctions[k]->Train(states, values);
  }
  printf("Done\n");

}
