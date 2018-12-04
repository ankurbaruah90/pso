#include <iostream>
#include <pso.h>

int main()
{
    PSO search;
    std::vector < std::vector <float> > parameters;
    parameters = search.getOptimal();

    for (std::vector < std::vector <float> >::iterator it = parameters.begin(); it != parameters.end(); ++it)
        std::cout << (*it)[1] << std::endl;

    return 0;
}
