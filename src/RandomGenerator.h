#ifndef RandomGenerator_h
#define RandomGenerator_h

#include <random>

class RandomGenerator
{
  public:

    static std::mt19937 engine;
    static bool isInitialized;

    inline static void Initialize()
    {
      if (!RandomGenerator::isInitialized) {
        std::random_device random;
        RandomGenerator::engine = std::mt19937(random());
        RandomGenerator::isInitialized = true;
      }
    }

    inline static double GetNormal()
    {
      RandomGenerator::Initialize();
      std::normal_distribution<double> normalDist;
      return normalDist(RandomGenerator::engine);
    }

    inline static double GetUniform()
    {
      RandomGenerator::Initialize();
      std::uniform_real_distribution<double> uniformDist;
      return uniformDist(RandomGenerator::engine);
    }

};

#endif // ifndef RandomGenerator_h
