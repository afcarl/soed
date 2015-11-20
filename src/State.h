#ifndef State_h
#define State_h

class State
{

public:

  virtual double GetMean() const;
  virtual double GetVariance() const;
  virtual std::pair<double, double> GetMoments() const;
  virtual double GetKL(std::shared_ptr<State> other) const;
  virtual std::shared_ptr<State> GetNextState(std::shared_ptr<Model>, const double control, const double disturbance);

};

#endif // ifndef State_h
