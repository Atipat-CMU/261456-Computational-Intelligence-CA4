#ifndef PARTICLE_H
#define PARTICLE_H

#include "../../mlp/object/Network.h"

namespace swarin {
    class Particle
    {
    private:
        Network *neuralN;
        double pbest;
        Parameter pbest_param;
        Parameter prev_v;

        double random(double min, double max);

    public:
        Particle(vector<layer_info> layers);
        ~Particle();

        double getValue(Dataframe X, Dataframe y);
        Parameter getParameter();
        void move(Parameter gbest_param, double c1, double c2);
    };
    
    Particle::Particle(vector<layer_info> layers)
    {
        this->neuralN = new Network(layers);
        this->pbest_param = neuralN->getParam();
        this->prev_v = Parameter(layers, 0, 0);
    }
    
    Particle::~Particle()
    {
        delete this->neuralN;
    }

    double Particle::random(double min, double max){
        float r1 = (float)rand() / (float)RAND_MAX;
        return min + r1 * (max - min);
    }

    double Particle::getValue(Dataframe X, Dataframe y){
        double value = this->neuralN->getError(X, y);
        if(value < this->pbest){
            this->pbest = value;
            this->pbest_param = this->neuralN->getParam();
        }
        return value;
    }

    Parameter Particle::getParameter(){
        return this->neuralN->getParam();
    }

    void Particle::move(Parameter gbest_param, double c1, double c2){
        double r1 = random(0,1);
        double r2 = random(0,1);
        double p1 = r1*c1;
        double p2 = r2*c2;
        Parameter x = this->neuralN->getParam();
        Parameter v = prev_v + (p1 * (this->pbest_param + (-1 * x))) + (p2 * (gbest_param + (-1 * x)));
        Parameter x_new = x + v;
        
        Parameter *param = new Parameter(x_new.get_weight_lys(), x_new.get_bias_lys());
        this->neuralN->setParam(param);
    }
    
}

#endif
