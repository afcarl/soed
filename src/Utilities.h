#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <Eigen/Core>

template<typename T>
inline T GetOption(const int argc, char **argv, const std::string& option, const T defaultOption)
{
  char **begin = argv;
  char **end   = argv + argc;
  char **itr   = std::find(begin, end, option);
  if ((itr != end) && (++itr != end)) {
    std::stringstream ss(*itr);
    T result;
    return (ss >> result) ? result : defaultOption;
  }
  return defaultOption;
}
inline double LogSumExp(const Eigen::VectorXd& v)
{
  return v.maxCoeff() + log(exp(v.array() - v.maxCoeff()).array().sum());
}
