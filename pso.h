#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <gsl/gsl_rng.h>
#include <string>
#include <time.h>
#include <deque>

#define FINETUNE 1

using namespace std;

#define V_MAX 0.05                                                              // max particle velocity
#define w 0.729                                                                 // inertia parameter w
#define c1 1.494                                                                // PSO parameter c1
#define c2 1.494                                                                // PSO parameter c2
#define SPAN 2                                                                  // Span of the swarm

/*---------------------------------*/
        /*---Add VNS---*/
/*---------------------------------*/

class PSO
{
public:
    struct tm tstruct;
    char log[100], amcl[100], termination[100];
    time_t now;
    const gsl_rng_type *T;
    gsl_rng * r;
    int swarmSize, iteration, dim, optimalSolution, epoch, solutionCount;
    float penalty, globalBestFitnessValue;
    float random1, random2;
    deque <float> bestGlobalFitnessQueue;
    float terminationThreshold;
    float bestFitness;
    int terminationWindow;

    ros::NodeHandle send, getError;
    ros::Publisher parameterPublish;
    vector <float> currentFitness, localBestFitness;
    vector < vector <float> > currentPosition, velocity, localBestPosition, globalBestPosition;

    void getFitness();
    void updateVelocityAndPosition();
    void initialize();
    void getOptimal();
    bool computeTerminationCondition();
    PSO();
    ~PSO();
};

PSO::PSO()
{
    now = time(0);
    tstruct = *localtime(&now);
    strftime(log, sizeof(log), "/home/ankur/catkin_ws/src/navigation/amcl/examples/Logs/log-%Y-%m-%d-%H-%M-%S", &tstruct);
    strcat(log, ".csv");
    strftime(amcl, sizeof(amcl), "/home/ankur/catkin_ws/src/navigation/amcl/examples/Logs/amcl-%Y-%m-%d-%H-%M-%S", &tstruct);
    strcat(amcl, ".csv");
    strftime(termination, sizeof(termination), "/home/ankur/catkin_ws/src/navigation/amcl/examples/Logs/term-%Y-%m-%d-%H-%M-%S", &tstruct);
    strcat(termination, ".csv");
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    random1 = 1.0;
    random2 = 1.0;
    swarmSize = 20;                                                             // size of swarm
    iteration = 10;                                                             // number of iterations
    dim = 6;                                                                    // dimensions/parameters
    epoch = 10;
    penalty = 100;
    globalBestFitnessValue = 10000.0;

    //Termination related parameters
    terminationWindow = 3;
    terminationThreshold = 0.02;
    bestFitness = 0.05;

    parameterPublish = send.advertise<std_msgs::Bool>("parameters", 1);
}

PSO::~PSO()
{

}

bool PSO::computeTerminationCondition()
{
    bestGlobalFitnessQueue.push_back(globalBestFitnessValue);
    if(bestGlobalFitnessQueue.size() > terminationWindow)
        bestGlobalFitnessQueue.pop_front();

    float stddev = 10000.0, devParticles = 1000.0;
    float mean = 0.0, meanParticles = 0.0;
    float sumofsquares = 0.0, sumSquareParticles;

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

    for (int j = 0; j < currentFitness.size(); j++)
        meanParticles += currentFitness[j];
    meanParticles = meanParticles/currentFitness.size();

    for (int j = 0; j < currentFitness.size(); j++)
        sumSquareParticles += pow((currentFitness[j] - meanParticles), 2);

    devParticles = sqrt(sumSquareParticles/currentFitness.size());

    //////////////////////////////////

    std::ofstream dataLog;
    ROS_INFO("Opening termination Log at %s $$$$$$$$", termination);
    dataLog.open(termination, std::ofstream::out | std::ofstream::app);

    for(int i = 0; i < bestGlobalFitnessQueue.size(); i++)
    {
        ROS_INFO("Fitness value %d %f", i, bestGlobalFitnessQueue[i]);
        dataLog << bestGlobalFitnessQueue[i] << ",";
    }

    ROS_INFO("Std dev : %f", stddev);
    dataLog << stddev << "," << meanParticles << "," << devParticles;
    dataLog << "\n";
    dataLog.close();

    if((stddev <= terminationThreshold) && (mean <= bestFitness))
        return true;
    else
        return false;
}

void PSO::getFitness()
{
    std_msgs::Bool paramDone;
    for (int i = 0; i < swarmSize; i++)
    {
        /* ------ pass parameters and receive fitness from device ------ */

        const char *path = "/home/ankur/catkin_ws/src/navigation/amcl/examples/parameters.yaml";
        std::ofstream text(path);
        if (text.is_open())
        {
            text << "odom_alpha1: " << currentPosition[0][i] << "\n"
                 << "odom_alpha2: " << currentPosition[1][i] << "\n"
                 << "odom_alpha3: " << currentPosition[2][i] << "\n"
                 << "odom_alpha4: " << currentPosition[3][i] << "\n"
                 << "update_min_a: " << currentPosition[4][i] << "\n"
                 << "update_min_d: " << currentPosition[5][i];
            text.close();
        }
        else
            cout << "Unable to open file\n";

        paramDone.data = true;
        parameterPublish.publish(paramDone);

        /* ---------- Subscribing Fitness ------------- */

        std_msgs::Float32::ConstPtr msg = ros::topic::waitForMessage <std_msgs::Float32> ("/Error", getError);        // subscribe error value
        currentFitness[i] = abs(msg->data);


        ///*---------Exterior Penalty - Quadratic Loss Function---------*///

        for (int k = 0; k < dim; k++)
            currentFitness[i] += penalty*(pow(fmin(0, currentPosition[k][i]),2));

        ///* ---------- Log ------------- *///

        std::ofstream textLog;
        textLog.open(log, std::ofstream::out | std::ofstream::app);
        textLog << msg->data << " , " << currentFitness[i] << "\t" << ",";
        textLog.close();

        cout << "Error " << msg->data << " Fitness " << currentFitness[i] << "\n";


        ///* ---------- Log, optimal parameters every iteration ------------- *///

        std::ofstream dataLog;
        dataLog.open(amcl, std::ofstream::out | std::ofstream::app);
        dataLog << i << "," << currentPosition[0][i] << "," << currentPosition[1][i] << ","
                << currentPosition[2][i] << "," << currentPosition[3][i] << ","
                << currentPosition[4][i] << "," << currentPosition[5][i] << ","
                << currentFitness[i] << "\t" << ",";
        dataLog.close();

        paramDone.data = false;
    }
    cout << "\n";
}

void PSO::updateVelocityAndPosition()
{
    double variableVelocity = 0;
    /* ------- update velocity and position ------ */
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < swarmSize; j++)
        {

#if !FINETUNE
            random1 = gsl_rng_uniform(r);
            random2 = gsl_rng_uniform(r);
#endif

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
#if FINETUNE
            std::ifstream file("/home/ankur/catkin_ws/src/navigation/amcl/examples/parameters.yaml");
            std::string str, value;
            std::vector <std::string> file_contents;
            while (std::getline(std::getline(file, str, ':'), value, '\n'))
                file_contents.push_back(value);
            double generateRandom = 0.2 * (gsl_rng_uniform(r) - 0.5);
            currentPosition[i][j] = generateRandom + atof(file_contents[i].c_str());
            if(currentPosition[i][j] < 0.0)
                currentPosition[i][j] = 0.001;
//            cout << "currentValue: " << currentPosition[i][j] << " originalValue: " << atof(file_contents[i].c_str()) << " difference: " << (currentPosition[i][j] - atof(file_contents[i].c_str())) << "\n";
            file.close();
#else
            currentPosition[i][j] = SPAN * (gsl_rng_uniform(r));         // random initial swarm positions, positive values
#endif
            velocity[i][j] = 0.05 * (gsl_rng_uniform(r));             // random initial swarm velocities
        }
    }
    localBestPosition = currentPosition;
}

void PSO::getOptimal()
{
    std::ofstream textLog, dataLog;
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
    for (int loop = 0; loop < epoch; loop++)
    {
        ///* ---------- Log ------------- *///
        textLog.open(log, std::ofstream::out | std::ofstream::app);
        textLog << "\nepoch: " << loop << " rate: " << penalty << "\n";
        textLog.close();

        dataLog.open(amcl, std::ofstream::out | std::ofstream::app);
        dataLog << "\nepoch: " << loop << " rate: " << penalty << "\n";
        dataLog.close();
        ///* ---------------------------- *///

        for (int i = 0; i < iteration; i++)
        {
            ///* ---------- Log ------------- *///
            textLog.open(log, std::ofstream::out | std::ofstream::app);
            textLog << "\niteration: " << i << "\n";
            textLog.close();

            dataLog.open(amcl, std::ofstream::out | std::ofstream::app);
            dataLog << "\niteration: " << i << "\n";
            dataLog.close();
            ///* ---------------------------- *///

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

            ///* ---------- Log ------------- *///
            dataLog.open(amcl, std::ofstream::out | std::ofstream::app);
            dataLog << globalBestPosition[0][1] << "," << globalBestPosition[1][1] << ","
                    << globalBestPosition[2][1] << "," << globalBestPosition[3][1] << ","
                    << globalBestPosition[4][1] << "," << globalBestPosition[5][1];
            dataLog.close();

            textLog.open(log, std::ofstream::out | std::ofstream::app);
            textLog << "Fitness: " << "," << globalBestFitnessValue;
            textLog.close();
            ///* ---------------------------- *///

            terminationConditionAchieved = computeTerminationCondition();
            if (terminationConditionAchieved == true)
                break;
            updateVelocityAndPosition();
        }
        if (terminationConditionAchieved == true)
            break;
        penalty *= 1;
    }
}

int main(int argc, char **argv)
{
    gsl_rng_env_setup();
    ros::init(argc, argv, "psotest");
    PSO object;
    object.getOptimal();
    return 0;
}
