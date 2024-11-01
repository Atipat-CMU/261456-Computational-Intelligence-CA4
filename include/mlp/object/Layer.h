#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <iostream>
#include <unordered_map>

using namespace std;

#include "Neural.h"
#include "Edge.h"
#include "Parameter.h"

using namespace mlp;

typedef unordered_map<Neural*, vector<Edge*>> edge_set;

namespace mlp {

    class Layer 
    {
        private:
            vector<Neural*> neurals;
            vector<Neural*> bias_ls;
            edge_set in_edges;
            edge_set out_edges;
            bool is_hidden;
            int ly_id;
            
            void clear();
            void copyFrom(const Layer& other);

        public:
            Layer();
            Layer(int ly_id, bool is_hidden, int N_node, double (*activation)(double, bool));
            ~Layer();

            Layer(const Layer& other);
            Layer& operator=(const Layer& other);

            int size();

            vector<Neural*> get_neurals() const;
            void connect(Layer* prev_ly, Parameter* param);
            void set_out_edges(edge_set out_edges);
            void set_input(vector<double>& inputs);
            void forward();
            void updateGrad(vector<double>& outputs);
            void backprop(double lr, double momentum);
            vector<double> get_output();
            void pull_param(Parameter* param);
            void push_param(Parameter* param);
    };

    Layer::Layer() {}

    Layer::Layer(int ly_id, bool is_hidden, int N_node, double (*activation)(double, bool)) {
        this->ly_id = ly_id;
        this->is_hidden = is_hidden;
        for(int i = 0; i < N_node; i++){
            Neural* neural = new Neural(is_hidden, activation);
            neurals.push_back(neural);
        }
    }

    Layer::~Layer(){
        for (Neural* neural : neurals) {
            if (neural != nullptr) {
                delete neural;
                neural = nullptr;
            }
        }
        neurals.clear();
    }

    Layer::Layer(const Layer& other) {
        clear();
        copyFrom(other);
    }

    Layer& Layer::operator=(const Layer& other) {
        if (this != &other) {
            clear();
            copyFrom(other);
        }
        return *this;
    }

    void Layer::clear() {
        for (Neural* neural : neurals) {
            delete neural;
        }
        neurals.clear();
        for (Neural* bias : bias_ls) {
            delete bias;
        }
        bias_ls.clear();
        for (auto& entry : in_edges) {
            for (Edge* edge : entry.second) {
                delete edge;
            }
        }
        in_edges.clear();
        for (auto& entry : out_edges) {
            for (Edge* edge : entry.second) {
                delete edge;
            }
        }
        out_edges.clear();
    }

    void Layer::copyFrom(const Layer& other) {
        for (Neural* neural : other.neurals) {
            neurals.push_back(new Neural(*neural));
        }
        for (Neural* bias : other.bias_ls) {
            bias_ls.push_back(new Neural(*bias));
        }
        for (const auto& entry : other.in_edges) {
            vector<Edge*> edges;
            for (Edge* edge : entry.second) {
                edges.push_back(new Edge(*edge));
            }
            in_edges[entry.first] = edges;
        }
    }

    int Layer::size() {
        return neurals.size();
    }

    vector<Neural*> Layer::get_neurals() const {
        return neurals;
    }

    void Layer::connect(Layer* prev_ly, Parameter* param) {
        int weight_count = 0;
        int bias_count = 0;
        for (Neural* nl : neurals) {
            vector<Edge*> edges_in;
            edge_set edges_out;

            for(Neural* prev_nl : prev_ly->get_neurals()){
                double weight = param->get_weight_ly(this->ly_id)[weight_count];
                Edge* edge = new Edge(nl, prev_nl, weight);
                edges_in.push_back(edge);
                edges_out[prev_nl].push_back(edge);
                weight_count++;
            }

            Neural* bias = new Neural(1);
            bias_ls.push_back(bias);
            double weight = param->get_bias_ly(this->ly_id)[bias_count];
            Edge* edge = new Edge(nl, bias, weight);
            edges_in.push_back(edge);
            bias_count++;

            in_edges[nl] = edges_in;
            prev_ly->set_out_edges(edges_out);
        }
    }

    void Layer::set_out_edges(edge_set out_edges) {
        this->out_edges = out_edges;
    }

    void Layer::set_input(vector<double>& inputs) {
        for(int i = 0; i < inputs.size(); i++){
            neurals[i]->setY(inputs[i]);
        }
    }

    vector<double> Layer::get_output() {
        this->forward();
        vector<double> outputs;
        for(int i = 0; i < neurals.size(); i++){
            outputs.push_back(neurals[i]->getY());
        }
        return outputs;
    }

    void Layer::pull_param(Parameter* param) {
        vector<double> weight_ls;
        vector<double> bias_ls;
        for(Neural* n : neurals){
            vector<Edge*> edges = in_edges[n];
            for(Edge* e : edges){
                weight_ls.push_back(e->getW());
            }
            bias_ls.push_back(weight_ls.back());
            weight_ls.pop_back();
        }
        
        param->set_weight_ly(ly_id, weight_ls);
        param->set_bias_ly(ly_id, bias_ls);
    }

    void Layer::push_param(Parameter* param) {
        vector<double> weight_ls = param->get_weight_ly(ly_id);
        vector<double> bias_ls = param->get_bias_ly(ly_id);
        int w_c = 0, b_c = 0;
        for(Neural* n : neurals){
            vector<Edge*> edges = in_edges[n];
            for(int i = 0; i < edges.size() - 1; i++){
                edges[i]->setW(weight_ls[w_c]);
                w_c++;
            }
            edges[edges.size() - 1]->setW(bias_ls[b_c]);
            b_c++;
        }
        
        param->set_weight_ly(ly_id, weight_ls);
        param->set_bias_ly(ly_id, bias_ls);
    }

    void Layer::forward() {
        for(Neural* n : neurals){
            vector<Edge*> edges = in_edges[n];
            double v = 0;
            for(Edge* e : edges){
                v += e->getTail()->getY() * e->getW();
            }
            n->update(v);
        }
    }

    void Layer::updateGrad(vector<double>& outputs) {
        if(is_hidden){
            for(int i = 0; i < neurals.size(); i++){
                double sum = 0;
                for(Edge* e : out_edges[neurals[i]]){
                    sum += e->getHead()->getG() * e->getW();
                }
                neurals[i]->updateGradHidden(sum);
            }
        } else {
            for(int i = 0; i < neurals.size(); i++){
                neurals[i]->updateGradOutput(outputs[i]);
            }
        }
    }

    void Layer::backprop(double lr, double momentum) {
        for(int i = 0; i < neurals.size(); i++){
            for(Edge* e : in_edges[neurals[i]]){
                double deltaW = lr * neurals[i]->getG() * e->getTail()->getY();
                e->setW(deltaW, momentum);
            }
        }
    }
}

#endif
