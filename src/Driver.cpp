#include <stuff>

int main(int argc, char** argv) {

  int stages = 3;
  
  auto model = make_shared<MossbauerExperiment>();
  
  auto solver = make_shared<DPSolver>(stages, model);
  
  vector<shared_ptr<State>> states(stages);
  vector<double> controls(stages);
  vector<double> costToGo(stages);
  vector<double> disturbances(stages);
  
  states[0] = model->GetPriorState();
  
  solver->Solve(states[0]);
  
  // now we execute the policy on some data
  
  double trueTheta = 0.5;
  
  for (int k = 0; k < stages - 1; ++k) {
    auto controlPair = solver->GetOptimalControl(states[k]);
    controls[k] = controlPair.first;
    costToGo[k] = controlPair.second;
    disturbances[k] = model->GetDisturbance(trueTheta, controls[k]);
    states[k + 1] = states[k]->GetNextState(model, controls[k], disturbances[k]); 
  }
  
  return 0;
  
}

