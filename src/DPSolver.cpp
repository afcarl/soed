#include <cstdio>
#include <iostream>
#include <cassert>
#include <Eigen/Core>
#include "DPSolver.h"

using namespace std;
using namespace Eigen;

VectorXd DPSolver::GetControls()
{
  return VectorXd::LinSpaced(numGridpoints, -1, 1);
}

std::pair<double, double> DPSolver::GetControlPair(const Eigen::VectorXd& values)
{
  VectorXd::Index optimalIndex;
  VectorXd controls = GetControls();
  double optimalValue = values.maxCoeff(&optimalIndex);
  double optimalControl = controls(optimalIndex);
  return make_pair(optimalControl, optimalValue);
}

void DPSolver::SetValueFunctionCoefficients(const MatrixXd& coefficients, shared_ptr<State> prior)
{
  assert(coefficients.cols() == numStages - 1);
  valueFunctions.resize(numStages);
  for (int k = 0; k < numStages - 1; ++k) {
    valueFunctions[k] = make_shared<ValueFunction>();
    valueFunctions[k]->coefficients = coefficients.col(k);
  }
  valueFunctions[numStages - 1] = make_shared<TerminalValueFunction>(prior);
}

// returns optimal control and value
VectorXd DPSolver::GetOptimalValues(shared_ptr<State> state, shared_ptr<ValueFunction> nextValueFunction)
{
  // enumerate all controls and values on a grid
  VectorXd controls = GetControls();
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
  return values;
}

VectorXd DPSolver::GetOptimalValues(shared_ptr<State> state, const int stage)
{
  return GetOptimalValues(state, valueFunctions[stage + 1]);
}

// solves Bellman's equation
void DPSolver::Solve(shared_ptr<State> prior)
{
  // reset value functions
  valueFunctions.resize(numStages);
  for (int i = 0; i < numStages - 1; ++i)
    valueFunctions[i] = make_shared<ValueFunction>();
  valueFunctions[numStages - 1] = make_shared<TerminalValueFunction>(prior);
  // compute training points
  VectorXd controls = GetControls();
  VectorXd priorSamples(numTrajectories);
  vector<vector<shared_ptr<State>>> trajectories(numTrajectories); // for each prior sample, compute trajectory of states
  #pragma omp parallel for ordered schedule(dynamic)
  for (int i = 0; i < numTrajectories; ++i) {
    trajectories[i].push_back(prior->GetCopy());
    for (int k = 1; k < numStages; ++k) {
      double control = controls(RandomGenerator::GetInt(0, numGridpoints - 1));
      double disturbance = model->GetDisturbance(priorSamples(i), control);
      trajectories[i].push_back(trajectories[i][k - 1]->GetNextState(model, control, disturbance));
    }
  }

  printf("Training value functions\n");
  for (int k = numStages - 2; k >= 1; --k) {
    printf("Computing values for stage k = %d\n", k);
    vector<shared_ptr<State>> states(numTrajectories);
    VectorXd values(numTrajectories);
    #pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < numTrajectories; ++i) {
      printf("%d/%d\r", i+1, numTrajectories); fflush(stdout);
      auto controlValues = GetOptimalValues(trajectories[i][k], valueFunctions[k + 1]);
      states[i] = trajectories[i][k];
      values(i) = controlValues.maxCoeff();
    }
    valueFunctions[k]->Train(states, values);
  }
  printf("\nDone\n");

}

VectorXd DPSolver::GetGreedyValues(shared_ptr<State> state)
{
  // enumerate all controls and values on a grid
  VectorXd controls = GetControls();
  VectorXd values = VectorXd::Zero(numGridpoints);
  VectorXd theta(numExpectation);
  VectorXd noise(numExpectation);
  for (int i = 0; i < numExpectation; ++i) {
    theta(i) = state->GetSample();
    noise(i) = model->GetNoiseSample();
  }
  #pragma omp parallel for ordered schedule(dynamic)
  for (int i = 0; i < numGridpoints; ++i) {
    printf("%d/%d\r", i+1, numGridpoints); fflush(stdout);
    double value = 0;
    for (int j = 0; j < numExpectation; ++j) {
      auto disturbance = model->GetDisturbance(theta(j), controls(i), noise(j));
      auto nextState = state->GetNextState(model, controls(i), disturbance);
      value += nextState->GetKL(state);
    }
    values(i) = value / numExpectation;
  }
  printf("\nDone\n");
  return values;
}