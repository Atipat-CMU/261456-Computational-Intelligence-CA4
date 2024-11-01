#include <iostream>
#include "../include/dotlis.h"
#include "../include/mlp.h"
#include "../include/swarin.h"

using namespace std;

using namespace swarin;

int main(){
    srand(time(0));

    Dataframe df = read_csv("AirQualityUCI.csv", 1);
    df = df.split_train_test({0.9}).first;

    vector<layer_info> layers = {
        {INPUT, nullptr, 8},
        {HIDDEN, linear, 10},
        {HIDDEN, linear, 10},
        {OUTPUT, linear, 1},
    };

    Dataframe X_train = df.get_column_without({1});
    Dataframe y_train = df.get_column({1});

    Swarm networks(layers, 60);

    networks.setData(X_train, y_train);

    int max = INT_MAX;
    int current = 0;
    for(int generation = 0; generation < 1000; generation++){
        current = networks.getError();
        if(current < max){
            max = current;
            networks.getBestParam().to_file("10-10-new60.param");
        }
        cout << "time " << generation << ": " << current << endl;
        networks.move();
    }
    
    return 0;
}
