#ifndef WEIGHTLIST_H
#define WEIGHTLIST_H

#include <vector>
#include <random>
#include <ctime>

using namespace std;

#include "LayerInfo.h"

namespace mlp {
    class Parameter
    {
        private:
            vector<vector<double>> weight_lys;
            vector<vector<double>> bias_lys;
            
        public:
            Parameter();
            Parameter(vector<layer_info> layers);
            Parameter(vector<layer_info> layers, double min, double max);
            Parameter(vector<vector<double>> weight_lys, vector<vector<double>> bias_lys);
            ~Parameter();

            vector<vector<double>> get_weight_lys();
            vector<vector<double>> get_bias_lys();
            vector<double> get_weight_ly(int ly);
            vector<double> get_bias_ly(int ly);
            vector<double> get_weight_unit(vector<layer_info> layers, int ly, int unit);
            double get_bias_unit(vector<layer_info> layers, int ly, int unit);
            void set_weight_ly(int ly, vector<double> weight_ls);
            void set_bias_ly(int ly, vector<double> bias_ls);
            void to_file(string filename);

            Parameter operator+(const Parameter& other) const;
            Parameter operator*(double scalar) const;
            friend Parameter operator*(double scalar, const Parameter& param);
    };
    
    Parameter::Parameter()
    {
    }

    Parameter::Parameter(vector<layer_info> layers){
        for(int j = 0; j < layers.size(); j++){
            layer_info curr_ly = layers[j];
            vector<double> weight_ly;
            vector<double> bias_ly;
            if(curr_ly.type != INPUT){
                layer_info prev_ly = layers[j-1];
                for(int i = 0; i < curr_ly.N_node; i++){
                    double max = 1/sqrt(prev_ly.N_node);
                    double min = -1/sqrt(prev_ly.N_node);
                    for(int k = 0; k < prev_ly.N_node; k++){
                        float r1 = (float)rand() / (float)RAND_MAX;
                        weight_ly.push_back(min + r1 * (max - min));
                    }
                    float r2 = (float)rand() / (float)RAND_MAX;
                    bias_ly.push_back(r2 * 1);
                }
            }
            weight_lys.push_back(weight_ly);
            bias_lys.push_back(bias_ly);
        }
    }

    Parameter::Parameter(vector<layer_info> layers, double min, double max){
        for(int j = 0; j < layers.size(); j++){
            layer_info curr_ly = layers[j];
            vector<double> weight_ly;
            vector<double> bias_ly;
            if(curr_ly.type != INPUT){
                layer_info prev_ly = layers[j-1];
                for(int i = 0; i < curr_ly.N_node; i++){
                    for(int k = 0; k < prev_ly.N_node; k++){
                        srand(time(0));
                        float r1 = (float)rand() / (float)RAND_MAX;
                        weight_ly.push_back(min + r1 * (max - min));
                    }
                    float r2 = (float)rand() / (float)RAND_MAX;
                    bias_ly.push_back(min + r2 * (max - min));
                }
            }
            weight_lys.push_back(weight_ly);
            bias_lys.push_back(bias_ly);
        }
    }

    Parameter::Parameter(vector<vector<double>> weight_lys, vector<vector<double>> bias_lys){
        this->weight_lys = weight_lys;
        this->bias_lys = bias_lys;
    }
    
    Parameter::~Parameter()
    {
    }

    vector<vector<double>> Parameter::get_weight_lys(){
        return weight_lys;
    }

    vector<vector<double>> Parameter::get_bias_lys(){
        return bias_lys;
    }

    vector<double> Parameter::get_weight_ly(int ly){
        return weight_lys[ly];
    }

    vector<double> Parameter::get_bias_ly(int ly){
        return bias_lys[ly];
    }

    vector<double> Parameter::get_weight_unit(vector<layer_info> layers, int ly, int unit){
        int prev_ly_N = layers[ly-1].N_node;
        int start = unit * prev_ly_N;
        int end = start + prev_ly_N;
        vector<double> unit_w;
        for(int j = start; j < end; j++){
            unit_w.push_back(weight_lys[ly][j]);
        }
        return unit_w;
    }

    double Parameter::get_bias_unit(vector<layer_info> layers, int ly, int unit){
        return bias_lys[ly][unit];
    }

    void Parameter::set_weight_ly(int ly, vector<double> weight_ls){
        if(weight_lys[ly].size() == weight_ls.size()){
            weight_lys[ly] = weight_ls;
        }else{
            throw runtime_error("Weight size not match");
        }
    }

    void Parameter::set_bias_ly(int ly, vector<double> bias_ls){
        if(bias_lys[ly].size() == bias_ls.size()){
            bias_lys[ly] = bias_ls;
        }else{
            throw runtime_error("Bias size not match");
        }
    }

    void Parameter::to_file(string filename){
        ofstream myfile;
        myfile.open (filename);
        for(vector<double> weight_ls : weight_lys){
            for(double weight : weight_ls){
                myfile << weight << " ";
            }
            myfile << "\n";
        }
        myfile << "#\n";
        for(vector<double> bias_ls : bias_lys){
            for(double bias : bias_ls){
                myfile << bias << " ";
            }
            myfile << "\n";
        }
        myfile.close();
    }

    Parameter param_read(string filename){
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Could not open file");
        }

        string line;

        vector<vector<double>> weight_lys;
        vector<vector<double>> bias_lys;
        bool isWeightReading = true;
        
        while (getline(file, line)) {
            if(line == "#"){
                isWeightReading = false;
                continue;
            }
            stringstream ss(line);
            string param;
            vector<double> param_ly;
            while (getline(ss, param, ' ')) {
                param_ly.push_back(stod(param));
            }
            if(isWeightReading){
                weight_lys.push_back(param_ly);
            }else{
                bias_lys.push_back(param_ly);
            }
        }

        return Parameter(weight_lys, bias_lys);
    }

    Parameter Parameter::operator+(const Parameter& other) const {
        vector<vector<double>> new_weight_lys;
        vector<vector<double>> new_bias_lys;

        for (size_t i = 0; i < weight_lys.size(); ++i) {
            vector<double> weight_add;
            for (size_t j = 0; j < weight_lys[i].size(); ++j) {
                weight_add.push_back(weight_lys[i][j] + other.weight_lys[i][j]);
            }
            new_weight_lys.push_back(weight_add);
        }

        for (size_t i = 0; i < bias_lys.size(); ++i) {
            vector<double> bias_add;
            for (size_t j = 0; j < bias_lys[i].size(); ++j) {
                bias_add.push_back(bias_lys[i][j] + other.bias_lys[i][j]);
            }
            new_bias_lys.push_back(bias_add);
        }

        return Parameter(new_weight_lys, new_bias_lys);
    }

    Parameter Parameter::operator*(double scalar) const {
        vector<vector<double>> new_weight_lys;
        vector<vector<double>> new_bias_lys;

        for (const auto& weight_ly : weight_lys) {
            vector<double> scaled_weight_ly;
            for (double weight : weight_ly) {
                scaled_weight_ly.push_back(weight * scalar);
            }
            new_weight_lys.push_back(scaled_weight_ly);
        }

        for (const auto& bias_ly : bias_lys) {
            vector<double> scaled_bias_ly;
            for (double bias : bias_ly) {
                scaled_bias_ly.push_back(bias * scalar);
            }
            new_bias_lys.push_back(scaled_bias_ly);
        }

        return Parameter(new_weight_lys, new_bias_lys);
    }

    Parameter operator*(double scalar, const Parameter& param) {
        return param * scalar;
    }
}

#endif
