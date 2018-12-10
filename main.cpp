#include <iostream>
#include <pso.h>
#include <time.h>
#include <fstream>
#include <string.h>

struct tm tstruct;
char plot[100];
time_t now;

float PSO::getScore(std::vector<float> values)
{

    // do something here and return the fitness score
    float x = values[0];
    float y = values[1];
    float cost = fabs(pow(x,2) - y);

    std::ofstream dataLog;
    dataLog.open(plot, std::ofstream::out | std::ofstream::app);
    dataLog << x << ", " << y << ", " << cost << "\n";
    dataLog.close();

    return cost;
}

int main()
{
    now = time(0);
    tstruct = *localtime(&now);
    strftime(plot, sizeof(plot), "/home/ankur/Desktop/pso/log/PSO-SVM-%Y-%m-%d-%H-%M-%S", &tstruct);
    strcat(plot, ".csv");

    PSO search;
    std::vector <float> parameters;
    parameters = search.getOptimal();

    for (std::vector <float>::iterator it = parameters.begin(); it != parameters.end(); ++it)
        std::cout << "Optimal Parameters " << *it << std::endl;

    std::cout << "Fitness Score " << pow(parameters[0],2) - parameters[1] << std::endl;

    return 0;
}
