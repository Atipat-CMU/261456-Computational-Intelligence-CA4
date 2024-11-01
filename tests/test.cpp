#include <iostream>
#include "../include/dotlis.h"
#include "../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df = read_csv("AirQualityUCI.csv", 1);

    df = df.split_train_test({0.9}).second;

    vector<layer_info> layers = {
        {INPUT, nullptr, 8},
        {HIDDEN, linear, 10},
        {HIDDEN, linear, 10},
        {OUTPUT, linear, 1},
    };

    Dataframe X_test = df.get_column_without({1});
    Dataframe y_test = df.get_column({1});

    Parameter loaded_param = param_read("10-10-new60.param");

    Network network(layers);
    network.setParam(new Parameter(loaded_param.get_weight_lys(), loaded_param.get_bias_lys()));

    cout << "---------------------------------------------------------" << endl;
    cout << "MAE: " << network.getError(X_test, y_test) << endl;

    return 0;
}
