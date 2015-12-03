#include <memory>
#include <vector>
#include <Eigen/Core>

#include "MossbauerModel.h"
#include "State.h"
#include "DPSolver.h"
#include "Utilities.h"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {

  int numStages           = GetOption<int>(argc, argv, "-numStages", 4);
  int numTrajectories     = GetOption<int>(argc, argv, "-numTrajectories", 2000);
  int numParticles        = GetOption<int>(argc, argv, "-numParticles", 2000);
  int numGridpoints       = GetOption<int>(argc, argv, "-numGridpoints", 21);
  int numExpectation      = GetOption<int>(argc, argv, "-numExpectation", 1000);

  double priorMean        = GetOption<double>(argc, argv, "-priorMean", 0.0);
  double priorVariance    = GetOption<double>(argc, argv, "-priorVariance", 1.0);
  double noiseVariance    = GetOption<double>(argc, argv, "-noiseVariance", 0.01);

  double trueTheta        = GetOption<double>(argc, argv, "-trueTheta", 0.5);

  string prefix           = GetOption<string>(argc, argv, "-prefix", "test");
  string coefficientsFile = GetOption<string>(argc, argv, "-coefficientsFile", "");

  string algorithm        = GetOption<string>(argc, argv, "-algorithm", "dp");

  auto model = make_shared<MossbauerModel>();
  model->SetPriorMean(priorMean);
  model->SetPriorVariance(priorVariance);
  model->SetNoiseVariance(noiseVariance);

  auto solver = make_shared<DPSolver>();
  solver->SetNumStages(numStages);
  solver->SetModel(model);
  solver->SetNumTrajectories(numTrajectories);
  solver->SetNumGridpoints(numGridpoints);
  solver->SetNumExpectation(numExpectation);

  // create prior State
  auto prior = make_shared<State>(VectorXd::Zero(numParticles), VectorXd::Zero(numParticles));
  for (int i = 0; i < numParticles; ++i)
    prior->particles(i) = model->GetPriorSample();

  cout << "Using algorithm: " << algorithm << endl;

  // choose appropriate thing to do based on algorithm
  if (algorithm == "dp") {
    if (coefficientsFile.empty()) {
      cout << "Solving for value function coefficients" << endl;
      solver->Solve(prior);
    } else {
      cout << "Reading value function coefficients from " << coefficientsFile << endl;
      MatrixXd coefficients = ReadEigenBinaryFile(coefficientsFile);
      solver->SetValueFunctionCoefficients(coefficients, prior);
    }
  } else if (algorithm == "greedy") {

  } else if (algorithm == "naive") {

  } else {
    cout << "Unknown algorithm: " << algorithm << endl;
    return 1;
  }

  // execute the policy for some true theta

  vector<shared_ptr<State>> states(numStages);
  VectorXd controls(numStages - 1);
  VectorXd costsToGo(numStages);
  VectorXd disturbances(numStages);

  states[0] = prior;

  for (int k = 0; k < numStages - 1; ++k) {
    if (algorithm == "dp") {
      auto controlPair = solver->GetOptimalControl(states[k], k);
      controls[k]      = controlPair.first;
      costsToGo[k]     = controlPair.second;
    } else if (algorithm == "greedy") {
      auto controlPair = solver->GetGreedyControl(states[k]);
      controls[k]      = controlPair.first;
      costsToGo[k]     = controlPair.second;
    } else if (algorithm == "naive") {
      auto grid = VectorXd::LinSpaced(numStages - 1, -1, 1);
      controls[k] = grid(k);
    }
    cout << "controls[" << k << "] = " << controls[k] << endl;
    disturbances[k]  = model->GetDisturbance(trueTheta, controls[k]);
    states[k + 1]    = states[k]->GetNextState(model, controls[k], disturbances[k]);

    auto moments = states[k+1]->GetMoments();
    cout << "mean:     " << moments.first << endl;
    cout << "variance: " << moments.second << endl;
    double kldiv = states[k+1]->GetKL(prior);
    cout << "DKL: " << kldiv << endl;
  }

  // dump stuff to files
  WriteEigenBinaryFile(prefix + ".controls", controls);

  // concatenate state matrices
  MatrixXd particles(numParticles, numStages);
  MatrixXd weights(numParticles, numStages);
  for (int k = 0; k < numStages; ++k) {
    particles.col(k) = states[k]->particles;
    weights.col(k) = states[k]->logWeights.array().exp();
  }
  WriteEigenBinaryFile(prefix + ".particles", particles);
  WriteEigenBinaryFile(prefix + ".weights", weights);

  // value function coefficients and training data
  if (algorithm == "dp" && coefficientsFile.empty()) {
    MatrixXd coefficients(ValueFunction::basisFunctions.size(), numStages - 1);
    MatrixXd trainingMeans(numTrajectories, numStages - 1);
    MatrixXd trainingVariances(numTrajectories, numStages - 1);
    MatrixXd trainingValues(numTrajectories, numStages - 1);
    for (int k = 0; k < numStages - 1; ++k) {
      coefficients.col(k) = solver->valueFunctions[k]->coefficients;
      trainingMeans.col(k) = solver->valueFunctions[k]->trainingMeans;
      trainingVariances.col(k) = solver->valueFunctions[k]->trainingVariances;
      trainingValues.col(k) = solver->valueFunctions[k]->trainingValues;
    }
    WriteEigenBinaryFile(prefix + ".coefficients", coefficients);
    WriteEigenBinaryFile(prefix + ".trainingMeans", trainingMeans);
    WriteEigenBinaryFile(prefix + ".trainingVariances", trainingVariances);
    WriteEigenBinaryFile(prefix + ".trainingValues", trainingValues);
  }

  return 0;

}

