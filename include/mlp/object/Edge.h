#ifndef EDGE_H
#define EDGE_H

#include "Neural.h"

namespace mlp {
    class Edge
    {
        private:
            Neural *head, *tail;
            double weight;
            double prev_deltaW = 0;

        public:
            Edge();
            Edge(Neural *head, Neural *tail, double weight);
            ~Edge();

            Neural* getHead();
            Neural* getTail();
            double getW();
            void setW(double deltaW, double momentum);
            void setW(double W);
    };

    Edge::Edge()
    {
    }

    Edge::Edge(Neural *head, Neural *tail, double weight){
        this->head = head;
        this->tail = tail;
        this->weight = weight;
    }

    Edge::~Edge()
    {
    }

    Neural* Edge::getHead(){
        return head;
    }

    Neural* Edge::getTail(){
        return tail;
    }

    double Edge::getW(){
        return weight;
    }

    void Edge::setW(double deltaW, double momentum){
        this->weight += deltaW + (momentum*prev_deltaW);
        this->prev_deltaW = deltaW;
    }

    void Edge::setW(double W){
        this->weight = W;
    }
}

#endif
