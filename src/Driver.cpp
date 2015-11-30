#include <memory>
#include <vector>
#include <Eigen/Core>

#include "MossbauerModel.h"
#include "State.h"
#include "DPSolver.h"
#include "Utilities.h"

using namespace std;

int main(int argc, char** argv) {

  int numStages          = GetOption<int>(argc, argv, "-numStages", 3);
  int numTrainingSamples = GetOption<int>(argc, argv, "-numTrainingSamples", 100);
  int numParticles       = GetOption<int>(argc, argv, "-numParticles", 1000);
  int numGridpoints      = GetOption<int>(argc, argv, "-numGridpoints", 21);
  
  double priorMean       = GetOption<double>(argc, argv, "-priorMean", 0.0);
  double priorVariance   = GetOption<double>(argc, argv, "-priorVariance", 1.0);
  double noiseVariance   = GetOption<double>(argc, argv, "-noiseVariance", 0.01);
  
  double trueTheta       = GetOption<double>(argc, argv "-trueTheta", 0.5);
  
  auto model = make_shared<MossbauerModel>();
  model->SetPriorMean(priorMean);
  model->SetPriorVariance(priorVariance);
  model->SetNoiseVariance(noiseVariance);

  auto solver = make_shared<DPSolver>();
  solver->SetNumStages(numStages);
  solver->SetModel(model);
  solver->SetNumTrainingSamples(numTrainingSamples);
  solver->SetNumGridpoints(numGridpoints);
  
  vector<shared_ptr<State>> states(numStages);
  vector<double> controls(numStages);
  vector<double> costsToGo(numStages);
  vector<double> disturbances(numStages);
  
  // create prior State
  auto prior = make_shared<State>();
  for (int i = 0; i < numParticles; ++i)
    prior->AddParticle(model->GetPriorSample());
  
  states[0] = prior;
  
  // compute optimal policy
  solver->Solve(states[0]);
  
  // now execute the policy on some data generated using trueTheta
  for (int k = 0; k < numStages - 1; ++k) {
    auto controlPair = solver->GetOptimalControl(states[k]);
    controls[k]      = controlPair.first;
    costsToGo[k]     = controlPair.second;
    disturbances[k]  = model->GetDisturbance(trueTheta, controls[k]);
    states[k + 1]    = states[k]->GetNextState(model, controls[k], disturbances[k]); 
  }
  
  // dump stuff to files
  
  return 0;
  
}

