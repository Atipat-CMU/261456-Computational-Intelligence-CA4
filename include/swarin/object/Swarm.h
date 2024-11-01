#ifndef SWARM_H
#define SWARM_H

#include "../../mlp/object/Network.h"
#include "../../swarin/object/Particle.h"

namespace swarin {
    class Swarm
    {
    private:
        int time = 0;
        vector<Particle*> s;
        int number;
        Dataframe X, y;
        vector<layer_info> layers;

        double gbest = INT_MAX;
        Parameter gbest_param;

    public:
        Swarm(vector<layer_info> layers, int n);
        ~Swarm();

        void setData(Dataframe X, Dataframe y);
        void move();
        double getError();
        Parameter getBestParam();
    };
    
    Swarm::Swarm(vector<layer_info> layers, int n)
    {
        this->number = n;
        for(int i = 0; i < n; i++){
            Particle* part = new Particle(layers);
            s.push_back(part);
        }
        this->layers = layers;
        this->gbest_param = Parameter(layers);
    }
    
    Swarm::~Swarm()
    {
        for (Particle* part : s) {
            if (part != nullptr) {
                delete part;
                part = nullptr;
            }
        }
        s.clear();
    }

    void Swarm::move(){
        Network network(this->layers);
        network.setParam(new Parameter(gbest_param.get_weight_lys(), gbest_param.get_bias_lys()));
        gbest = network.getError(X, y);

        double c1 = 2, c2 = 1;
        for(Particle* part : s){
            double value = part->getValue(this->X, this->y);
            if(value < this->gbest){
                this->gbest = value;
                this->gbest_param = part->getParameter();
            }
        }
        for(Particle* part : s){
            part->move(this->gbest_param, c1, c2);
        }
    }

    void Swarm::setData(Dataframe X, Dataframe y){
        this->X = X;
        this->y = y;
    }

    double Swarm::getError(){
        Network network(this->layers);
        network.setParam(new Parameter(gbest_param.get_weight_lys(), gbest_param.get_bias_lys()));
        return network.getError(X, y);
    }

    Parameter Swarm::getBestParam(){
        return this->gbest_param;
    }
    
}

#endif
