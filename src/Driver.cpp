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

  int numStages          = GetOption<int>(argc, argv, "-numStages", 4);
  int numTrainingSamples = GetOption<int>(argc, argv, "-numTrainingSamples", 1000);
  int numParticles       = GetOption<int>(argc, argv, "-numParticles", 2000);
  int numGridpoints      = GetOption<int>(argc, argv, "-numGridpoints", 21);
  int numExpectation     = GetOption<int>(argc, argv, "-numExpectation", 500);

  double priorMean       = GetOption<double>(argc, argv, "-priorMean", 0.0);
  double priorVariance   = GetOption<double>(argc, argv, "-priorVariance", 1.0);
  double noiseVariance   = GetOption<double>(argc, argv, "-noiseVariance", 0.04);

  double trueTheta       = GetOption<double>(argc, argv, "-trueTheta", 0.5);

  string prefix          = GetOption<string>(argc, argv, "-prefix", "test");

  auto model = make_shared<MossbauerModel>();
  model->SetPriorMean(priorMean);
  model->SetPriorVariance(priorVariance);
  model->SetNoiseVariance(noiseVariance);

  auto solver = make_shared<DPSolver>();
  solver->SetNumStages(numStages);
  solver->SetModel(model);
  solver->SetNumTrainingSamples(numTrainingSamples);
  solver->SetNumGridpoints(numGridpoints);
  solver->SetNumExpectation(numExpectation);

  // create prior State
  auto prior = make_shared<State>(VectorXd::Zero(numParticles), VectorXd::Zero(numParticles));
  for (int i = 0; i < numParticles; ++i)
    prior->particles(i) = model->GetPriorSample();

  // compute optimal policy
  solver->Solve(prior);

  // execute the optimal policy on some synthetic data

  vector<shared_ptr<State>> states(numStages);
  VectorXd controls(numStages - 1);
  VectorXd costsToGo(numStages);
  VectorXd disturbances(numStages);

  states[0] = prior;

  for (int k = 0; k < numStages - 1; ++k) {
    auto controlPair = solver->GetOptimalControl(states[k], k);
    controls[k]      = controlPair.first;
    costsToGo[k]     = controlPair.second;
    disturbances[k]  = model->GetDisturbance(trueTheta, controls[k]);
    states[k + 1]    = states[k]->GetNextState(model, controls[k], disturbances[k]);
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

  // value function coefficients
  MatrixXd coefficients(ValueFunction::basisFunctions.size(), numStages - 1);
  for (int k = 0; k < numStages - 1; ++k) {
    coefficients.col(k) = solver->valueFunctions[k]->coefficients;
  }
  WriteEigenBinaryFile(prefix + ".coefficients", coefficients);

  return 0;

}

