#include <string>
#include <iostream>
#include <fstream>
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
  file.open(path, std::ios::out | std::ios::binary);
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

inline Eigen::MatrixXd ReadEigenBinaryFile(const std::string& path)
{
  std::ifstream file;
  file.open(path, std::ios::in | std::ios::binary);
  int rows, cols;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  Eigen::MatrixXd m(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double entry;
      file.read(reinterpret_cast<char *>(&entry), sizeof(double));
      m(i, j) = entry;
    }
  }
  file.close();
  return m;
}

