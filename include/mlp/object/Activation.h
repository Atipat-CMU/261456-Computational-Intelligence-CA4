#include <math.h>

namespace mlp {
    double sigmoid(double x, bool isForward){
        if(isForward){
            return 1/(1+exp(-x));
        }else{
            double f = sigmoid(x, true);
            return f*(1-f);
        }
    }

    double tanh(double x, bool isForward) {
        if (isForward) {
            return (2/(1+exp(-2*x))) - 1;
        } else {
            double f = tanh(x, true);
            return 1 - (f * f);
        }
    }

    double linear(double x, bool isForward){
        if(isForward){
            return x;
        }else{
            return 1;
        }
    }
}
