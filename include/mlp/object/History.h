#ifndef HISTORY_H
#define HISTORY_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

namespace mlp {
    class History
    {
        private:
            vector<double> error_ls;

        public:
            History();
            History(vector<double> error_ls);
            ~History();

            double get_latest_err();
            void exportError(string filename);
    };

    History::History()
    {
    }

    History::History(vector<double> error_ls){
        this->error_ls = error_ls;
    }

    History::~History()
    {
    }

    double History::get_latest_err(){
        return this->error_ls.back();
    }

    void History::exportError(string filename){
        ofstream myfile;
        myfile.open (filename);
        myfile << "epochs,error\n";
        for(int i = 0; i < error_ls.size(); i++){
            myfile << i << "," << error_ls[i] << "\n";
        }
        myfile.close();
    }
}

#endif
