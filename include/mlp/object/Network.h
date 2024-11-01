#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include <ctime>

using namespace std;

#include "Layer.h"
#include "../../dotlis/object/Dataframe.h"
#include "LayerInfo.h"
#include "Parameter.h"
#include "History.h"

using namespace dotlis;

namespace mlp {
    class Network
    {
        private:
            Layer *input_ly, *output_ly;
            vector<Layer*> hidden_lys;
            Parameter *parameter; // Changed to pointer
            void forward(vector<double>& inputs);
            void backward(vector<double>& outputs, double lr, double momentum);
            void update_param();

        public:
            Network();
            Network(vector<layer_info> layers);
            ~Network();

            void info();
            History fit(Dataframe X, Dataframe y, int epoch, double lr, double momentum);
            double getError(Dataframe X, Dataframe y);
            vector<double> predict_one(vector<double> input);
            Dataframe predict(Dataframe inputs);
            Parameter getParam();
            void setParam(Parameter *parameter);
    };

    Network::Network() : parameter(nullptr) {}

    Network::Network(vector<layer_info> layers) {
        parameter = new Parameter(layers);  // Allocate memory for parameter pointer
        int ly_count = 0;
        for(layer_info l_info : layers){
            Layer* layer = new Layer(ly_count, l_info.type == HIDDEN, l_info.N_node, l_info.activation);
            if(l_info.type == INPUT){
                input_ly = layer;
            } else if(l_info.type == OUTPUT){
                output_ly = layer;
            } else {
                hidden_lys.push_back(layer);
            }
            ly_count++;
        }

        if(!hidden_lys.empty()){
            hidden_lys[0]->connect(input_ly, parameter);
            for(int i = 1; i < hidden_lys.size(); i++){
                hidden_lys[i]->connect(hidden_lys[i-1], parameter);
            }
            output_ly->connect(hidden_lys[hidden_lys.size()-1], parameter);
        } else {
            output_ly->connect(input_ly, parameter);
        }
    }

    Network::~Network(){
        delete input_ly;
        delete output_ly;
        for (Layer* layer : hidden_lys) {
            delete layer;
        }
        delete parameter;  // Free memory for parameter
    }

    void Network::info(){
        for(int i = 1; i <= hidden_lys.size(); i++){
            cout << "layer " << i << " : " << hidden_lys[i-1] << endl;
        }
    }

    void Network::forward(vector<double>& inputs){
        input_ly->set_input(inputs);
        for(Layer* ly : hidden_lys){
            ly->forward();
        }
        output_ly->forward();
    }

    void Network::backward(vector<double>& outputs, double lr, double momentum){
        output_ly->updateGrad(outputs);
        for(auto it = hidden_lys.rbegin(); it != hidden_lys.rend(); ++it){
            (*it)->updateGrad(outputs);
        }
    
        output_ly->backprop(lr, momentum);
        for(auto it = hidden_lys.rbegin(); it != hidden_lys.rend(); ++it){
            (*it)->backprop(lr, momentum);
        }
    }

    void Network::update_param(){
        for(int i = 0; i < hidden_lys.size(); i++){
            hidden_lys[i]->pull_param(parameter);
        }
        output_ly->pull_param(parameter);
    }

    Parameter Network::getParam(){
        return *parameter;
    }

    void Network::setParam(Parameter *parameter){
        if (this->parameter != nullptr) {
            delete this->parameter; // Delete old parameter if exists
        }
        this->parameter = parameter;  // Assign new parameter pointer
        for(int i = 0; i < hidden_lys.size(); i++){
            hidden_lys[i]->push_param(parameter);
        }
        output_ly->push_param(parameter);
    }

    History Network::fit(Dataframe X, Dataframe y, int epoch, double lr, double momentum){
        if(X.get_width() != input_ly->size()){
            throw runtime_error("Input size not match");
        }
        if(y.get_width() != output_ly->size()){
            throw runtime_error("Output size not match");
        }

        vector<double> error_ls;

        while(epoch--){
            srand(time(0));

            vector<int> index_ls;
            for(int i = 0; i < X.get_depth(); i++){
                index_ls.push_back(i);
            }

            double error = 0;
            while(!index_ls.empty()){
                int range = (index_ls.size() - 1) + 1;
                int rnum = rand() % range;

                int index = index_ls[rnum];
                vector<double> inputs = X.getRow(index);
                vector<double> outputs = y.getRow(index);

                this->forward(inputs);

                vector<double> y_ls = output_ly->get_output();
                double sse = 0;
                for(int i = 0; i < y_ls.size(); i++){
                    sse += pow(outputs[i] - y_ls[i], 2);
                }

                error += sse / 2.0;

                this->backward(outputs, lr, momentum);
                index_ls.erase(index_ls.begin() + rnum);
            }

            error_ls.push_back(error / X.get_depth());
        }

        this->update_param();
        return History(error_ls);
    }

    double Network::getError(Dataframe X, Dataframe y){
        double error = 0;

        for(int i = 0; i < X.get_depth(); i++){
            vector<double> inputs = X.getRow(i);
            vector<double> outputs = y.getRow(i);

            this->forward(inputs);

            vector<double> y_ls = output_ly->get_output();
            double sse = 0;
            for(int i = 0; i < y_ls.size(); i++){
                sse += abs(outputs[i] - y_ls[i]);
            }

            error += sse;
        }

        return error / X.get_depth();
    }

    vector<double> Network::predict_one(vector<double> inputs){
        this->forward(inputs);
        return output_ly->get_output();
    }

    Dataframe Network::predict(Dataframe df_test){
        Dataframe df_predict;
        for(int i = 0; i < df_test.get_depth(); i++){
            df_predict.insert(predict_one(df_test.getRow(i)));
        }
        return df_predict;
    }
}

#endif
