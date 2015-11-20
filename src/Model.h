#ifndef Model_h
#define Model_h

class Model
{

public:

  virtual double GetLogLikelihood(double theta, double control, double disturbance) = 0;
  
  virtual double GetDisturbance(double theta, double control) = 0;
  
  virtual double GetPriorSample() = 0;

};

#endif // ifndef Model_h
