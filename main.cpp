#include <iostream>
#include <pso.h>

float PSO::getScore(std::vector<float> values)
{

    // do something here and return the fitness score

    return 0.0;
}

int main()
{
    PSO search;
    std::vector <float> parameters;
    parameters = search.getOptimal();

    for (std::vector <float>::iterator it = parameters.begin(); it != parameters.end(); ++it)
        std::cout << "Optimal Parameters " << *it << std::endl;

    return 0;
}
