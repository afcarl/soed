#ifndef MossbauerModel_h
#define MossbauerModel_h

#include "Model.h"

class MossbauerModel : public Model 
{

public:

  double Evaluate(const double theta, const double d);
  
  double GetLogLikelihood(double theta, double control, double disturbance) override;
  
  double GetDisturbance(double theta, double control) override;
  
  double GetPriorSample() override;

};

#endif // MossbauerModel_h
