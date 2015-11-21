#ifndef TerminalValueFunction_h
#define TerminalValueFunction_h

#include "ValueFunction.h"

class TerminalValueFunction : public ValueFunction
{

public:

  std::shared_ptr<State> prior;

  TerminalValueFunction(std::shared_ptr<State> prior) : prior(prior) { }
  
  inline double Evaluate(std::shared_ptr<const State> state) override
  {
    return state->GetKL(prior);
  }
  
  inline void Train(const std::vector<std::shared_ptr<const State>> states, const Eigen::VectorXd& costs) override 
  {
    // pass  
  }

};

#endif // ifndef TerminalValueFunction_h
