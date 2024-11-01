#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <cmath>

#include "Dataframe.h"

namespace dotlis{
    double calRMSE(Dataframe y, Dataframe _y){
        if(y.get_depth() != _y.get_depth() 
            || y.get_width() != 1 || y.get_width() != 1){
            runtime_error("input is invalid");
        }
        int N = y.get_depth();
        double sse = 0;
        for(int i = 0; i < N; i++){
            sse += pow(y.get(i,0) - _y.get(i,0), 2);
        }
        return sqrt(sse/N);
    }

    Dataframe markMax(Dataframe y){
        vector<vector<double>> table(y.get_depth(), vector<double>(y.get_width(), 0));
        for(int r = 0; r < y.get_depth(); r++){
            int max_idx = 0;
            double max = 0;
            for(int c = 0; c < y.get_width(); c++){
                if(y.get(r, c) > max){
                    max = y.get(r, c);
                    max_idx = c;
                }
            }
            table[r][max_idx] = 1;
        }
        return Dataframe(table);
    }

    double calConfusionM(Dataframe y, Dataframe _y){
        vector<vector<int>> table(2, vector<int>(2,0));
        if(y.get_depth() != _y.get_depth() 
            || y.get_width() != 1 || y.get_width() != 1){
            runtime_error("input is invalid");
        }
        int N = y.get_depth();
        double sse = 0;
        for(int i = 0; i < N; i++){
            table[y.get(i,0)][_y.get(i,0)]++;
        }

        int a = table[0][0], b = table[0][1];
        int c = table[1][0], d = table[1][1];

        cout << a << "  " << b << "\n";
        cout << c << "  " << d << "\n";

        cout << "Accuracy: " << (a+d)/(N*1.0) << "\t";
        cout << "Miss: " << (b+c)/(N*1.0) << endl;

        return (a+d)/(N*1.0);
    }
}

#endif
