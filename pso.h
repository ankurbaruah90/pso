#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <deque>

using namespace std;

#define V_MAX 0.05                                                              // max particle velocity
#define w 0.729                                                                 // inertia parameter w
#define c1 1.494                                                                // PSO parameter c1
#define c2 1.494                                                                // PSO parameter c2
#define SPAN 2                                                                  // Span of the swarm

class PSO
{
public:
    struct tm tstruct;
    time_t now;
    const gsl_rng_type *T;
    gsl_rng * r;
    int swarmSize, iteration, dim, optimalSolution, solutionCount;
    float penalty, globalBestFitnessValue;
    float random1, random2;
    deque <float> bestGlobalFitnessQueue;
    float terminationThreshold;
    float bestFitness;
    int terminationWindow;

    vector <float> currentFitness, localBestFitness;
    vector < vector <float> > currentPosition, velocity, localBestPosition, globalBestPosition;

    void getFitness();
    void updateVelocityAndPosition();
    void initialize();
    vector<vector<float> > getOptimal();
    bool computeTerminationCondition();
    PSO();
    ~PSO();
};

PSO::PSO()
{
    now = time(0);
    tstruct = *localtime(&now);
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    swarmSize = 20;                                                             // size of swarm
    iteration = 10;                                                             // number of iterations
    dim = 6;                                                                    // dimensions/parameters
    penalty = 100;
    globalBestFitnessValue = 10000.0;

    //Termination related parameters
    terminationWindow = 3;
    terminationThreshold = 0.02;
    bestFitness = 0.05;
}

PSO::~PSO()
{

}

//TODO: check for sanity of termination condition
bool PSO::computeTerminationCondition()
{
    bestGlobalFitnessQueue.push_back(globalBestFitnessValue);
    if(bestGlobalFitnessQueue.size() > terminationWindow)
        bestGlobalFitnessQueue.pop_front();

    float meanParticles[dim], sumSquareParticles;
    float stddev = 10000.0, devParticles = 1000.0;
    float mean = 0.0;
    float sumofsquares = 0.0;

    if(bestGlobalFitnessQueue.size() == terminationWindow)
    {
        for(int i = 0; i < bestGlobalFitnessQueue.size(); i++)
        {
            mean += bestGlobalFitnessQueue[i];
        }
        mean = mean/bestGlobalFitnessQueue.size();
        for(int i = 0; i < bestGlobalFitnessQueue.size(); i++)
            sumofsquares += pow(bestGlobalFitnessQueue[i] - mean, 2);
        stddev = sqrt(sumofsquares/bestGlobalFitnessQueue.size());
    }

    ///// Size of Particle Cloud /////
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < swarmSize; j++)
            meanParticles[i] += currentPosition[i][j];

    for (int k = 0; k < dim; k++)
        meanParticles[k] = meanParticles[k]/swarmSize;

    for (int i = 0; i < dim; i++)
        for (int j = 0; j < swarmSize; j++)
            sumSquareParticles += pow((currentPosition[i][j] - meanParticles[i]), 2);

    devParticles = sqrt(sumSquareParticles/swarmSize);

    if((stddev <= terminationThreshold) && (mean <= bestFitness))
        return true;
    else
        return false;
}

void PSO::getFitness()
{
    for (int i = 0; i < swarmSize; i++)
    {

        ///-----YOUR CODE GOES HERE-----///

        /* ---------- Subscribing Fitness ------------- */

//        std_msgs::Float32::ConstPtr msg = ros::topic::waitForMessage <std_msgs::Float32> ("/Error", getError);        // subscribe error value
//        currentFitness[i] = abs(msg->data);


        ///*---------Exterior Penalty - Quadratic Loss Function---------*///
        ///----- Add your custom penalty functions here------///

        for (int k = 0; k < dim; k++)
            currentFitness[i] += penalty*(pow(fmin(0, currentPosition[k][i]),2));

    }
}

void PSO::updateVelocityAndPosition()
{
    double variableVelocity = 0;
    /* ------- update velocity and position ------ */
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < swarmSize; j++)
        {
            // These random variables further stirs up the swarm
            // You might want to cross-verify the values here
            random1 = gsl_rng_uniform(r);
            random2 = gsl_rng_uniform(r);

            variableVelocity = w * velocity[i][j] + c1 * (random1 * (localBestPosition[i][j] - currentPosition[i][j])) + c2 * (random2 * (globalBestPosition[i][j] - currentPosition[i][j]));
            if (variableVelocity < 0)
                velocity[i][j] = fmax(variableVelocity, (-1.0 * V_MAX));
            else
                velocity[i][j] = fmin(variableVelocity, V_MAX);

            currentPosition[i][j] = currentPosition[i][j] + velocity[i][j];
            if(currentPosition[i][j] < 0.0)
                currentPosition[i][j] = 0.001;
        }
}

void PSO::initialize()
{
    /* ------------ Initialize Swarm  -------------  */
    localBestFitness.resize(swarmSize);
    currentFitness.resize(swarmSize);
    localBestPosition.resize(dim);
    globalBestPosition.resize(dim);
    currentPosition.resize(dim);
    velocity.resize(dim);
    for (int i = 0 ; i < dim; i++)
    {
        currentPosition[i].resize(swarmSize);
        localBestPosition[i].resize(swarmSize);
        globalBestPosition[i].resize(swarmSize);
        velocity[i].resize(swarmSize);
        for (int j = 0; j < swarmSize; j++)
        {
            currentPosition[i][j] = SPAN * (gsl_rng_uniform(r));         // random initial swarm positions, positive values
            velocity[i][j] = 0.05 * (gsl_rng_uniform(r));             // random initial swarm velocities
        }
    }
    localBestPosition = currentPosition;
}

vector <vector <float> > PSO::getOptimal()
{
    initialize();                                                               // Initialize Swarm
    getFitness();                                                               // pass parameters and get fitness
    bool terminationConditionAchieved = false;
    localBestFitness = currentFitness;
    vector <float>::iterator globalBestFitness = min_element(localBestFitness.begin(), localBestFitness.end());
    globalBestFitnessValue = localBestFitness[distance(localBestFitness.begin(), globalBestFitness)];

    for (int a = 0; a < swarmSize; a++)
        for (int b = 0; b < dim; b++)
            globalBestPosition[b][a] = localBestPosition[b][distance(localBestFitness.begin(), globalBestFitness)];

    updateVelocityAndPosition();

    /* --------------- Update Swarm -----------------  */
    for (int i = 0; i < iteration; i++)
    {

        getFitness();
        for (int j = 0; j < swarmSize; j++)
        {
            if (currentFitness[j] < localBestFitness[j])
            {
                localBestFitness[j] = currentFitness[j];
                for (int c = 0; c < dim; c++)
                    localBestPosition[c][j] = currentPosition[c][j];
            }
        }

        vector <float>::iterator currentGlobalBestFitness = min_element(localBestFitness.begin(), localBestFitness.end());
        float currentGlobalBestFitnessValue = localBestFitness[distance(localBestFitness.begin(), currentGlobalBestFitness)];
        if (currentGlobalBestFitnessValue < globalBestFitnessValue)
        {
            globalBestFitnessValue = currentGlobalBestFitnessValue;
            for (int a = 0; a < swarmSize; a++)
                for (int b = 0; b < dim; b++)
                    globalBestPosition[b][a] = localBestPosition[b][distance(localBestFitness.begin(), currentGlobalBestFitness)];
        }

        terminationConditionAchieved = computeTerminationCondition();
        if (terminationConditionAchieved == true)
            break;

        updateVelocityAndPosition();
    }
    return globalBestPosition;
}
