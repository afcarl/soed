#include <string>
#include <iostream>
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

inline void WriteEigenBinaryFile(const std::string& path, const Eigen::MatrixXd& m)
{
  std::ofstream file;
  file.open(path, ios::out | ios::binary);
  int rows = m.rows();
  int cols = m.cols();
  file.write(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<char *>(&cols), sizeof(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double entry = m(i, j);
      file.write(reinterpret_cast<char *>(&entry), sizeof(double));
    }
  }
  file.close();
}
