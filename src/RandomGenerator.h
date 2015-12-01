#ifndef RandomGenerator_h
#define RandomGenerator_h

#include <random>

class RandomGenerator
{
  public:

    static std::mt19937 engine;
    static bool isInitialized;
    static std::normal_distribution<double> normalDist;
    static std::uniform_real_distribution<double> uniformDist;

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
      return RandomGenerator::normalDist(RandomGenerator::engine);
    }

    inline static double GetUniform()
    {
      RandomGenerator::Initialize();
      return RandomGenerator::uniformDist(RandomGenerator::engine);
    }
    
};

#endif // ifndef RandomGenerator_h
