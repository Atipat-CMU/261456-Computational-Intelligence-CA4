#ifndef NEURAL_H
#define NEURAL_H

#include <vector>

namespace mlp {
    class Neural
    {
        private:
            bool is_hidden;
            double (*activation)(double, bool);
            double v, y = 0;
            double localGrad = 0;

        public:
            Neural();
            Neural(double y);
            Neural(bool is_hidden, double (*fun)(double, bool));
            ~Neural();

            void update(double v);
            void setY(double y);
            double getY();

            void updateGradOutput(double d);
            void updateGradHidden(double sum);
            double getG();
    };

    Neural::Neural()
    {
    }

    Neural::Neural(double y){
        this->y = y;
    }

    Neural::Neural(bool is_hidden, double (*fun)(double, bool)){
        this->is_hidden = is_hidden;
        this->activation = fun;
        this->y = 0;
        this->v = 0;
    }

    Neural::~Neural()
    {
    }

    void Neural::update(double v){
        this->v = v;
        this->y = activation(v, true);
    }

    void Neural::setY(double y){
        this->y = y;
    }

    double Neural::getY(){
        return y;
    }

    void Neural::updateGradOutput(double d){
        double error = d - y;
        this->localGrad = error*activation(v, false);
    }

    void Neural::updateGradHidden(double sum){
        this->localGrad = sum*activation(v, false);
    }

    double Neural::getG(){
        return localGrad;
    }
}

#endif
