#ifndef NORMALIZER_H
#define NORMALIZER_H

#include "Dataframe.h"

namespace dotlis {
    class Normalizer
    {
    private:
        double min_i, max_i;
        double min_o, max_o;

    public:
        Normalizer();
        Normalizer(double, double, double, double);
        ~Normalizer();

        Dataframe forward(Dataframe orig);
        Dataframe backward(Dataframe norm);
    };
    
    Normalizer::Normalizer()
    {
    }

    Normalizer::Normalizer(double min_i, double max_i, double min_o, double max_o){
        this->min_i = min_i;
        this->max_i = max_i;
        this->min_o = min_o;
        this->max_o = max_o;
    }
    
    Normalizer::~Normalizer()
    {
    }

    Dataframe Normalizer::forward(Dataframe orig){
        vector<vector<double>> table;
        for(int r = 0; r < orig.get_depth(); r++){
            vector<double> row;
            for(int c = 0; c < orig.get_width(); c++){
                double x = orig.get(r, c);
                row.push_back(((((x-min_i)/(max_i - min_i))*((max_o - min_o))) + min_o));
            }
            table.push_back(row);
        }
        return Dataframe(table);
    }

    Dataframe Normalizer::backward(Dataframe norm){
        vector<vector<double>> table;
        for(int r = 0; r < norm.get_depth(); r++){
            vector<double> row;
            for(int c = 0; c < norm.get_width(); c++){
                double x = norm.get(r, c);
                row.push_back(((((x-min_o)/(max_o - min_o))*(max_i - min_i)) + min_i));
            }
            table.push_back(row);
        }
        return Dataframe(table);
    }
    
}

#endif
