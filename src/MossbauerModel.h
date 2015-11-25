#ifndef MossbauerModel_h
#define MossbauerModel_h

#include <cmath>

#include "RandomGenerator.h"
#include "Model.h"

class MossbauerModel : public Model 
{

private:

  double priorMean;
  double priorVariance;
  double noiseVariance;

public:

  inline void SetPriorMean(const double priorMean) { this->priorMean = priorMean; }
  inline void SetPriorMean(const double priorMean) { this->priorMean = priorMean; }
  inline void SetPriorMean(const double priorMean) { this->priorMean = priorMean; }

  inline double Evaluate(const double theta, const double control) 
  {
    return 1.0 - 0.1 / ((theta - control) * (theta - control) + 0.1;
  }
  
  inline double GetLogLikelihood(double theta, double control, double disturbance) override
  {
    double output = Evaluate(theta, control);
    return -0.5 * log(2.0 * M_PI * noiseVariance) - (output - disturbance) * (output - disturbance) / (2.0 * noiseVariance);
  }
  
  inline double GetDisturbance(double theta, double control) override
  {
    return Evaluate(theta, control) + RandomGenerator::GetNormal() * sqrt(noiseVariance);
  }
  
  inline double GetPriorSample() override
  {
    return priorMean + RandomGenerator::GetNormal() * sqrt(priorVariance);
  }

};

#endif // MossbauerModel_h
