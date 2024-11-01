#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <regex>

using namespace std;

namespace dotlis {
    class Dataframe
    {
        private:
            vector<vector<double>> table;
            int width = 0;

        public:
            Dataframe();
            Dataframe(const string& file_path, int start_line);
            Dataframe(vector<vector<double>> table);
            ~Dataframe();

            void to_csv(string filename);

            int get_width() const;
            int get_depth() const;
            double get(int r, int c) const;
            vector<double> getRow(int r) const;
            void insert(vector<double> row);

            Dataframe get_column(const vector<int>& columns) const;
            Dataframe get_column_without(const vector<int>& columns) const;
            Dataframe get_onehot(const int& column) const;
            vector<Dataframe> split_n(int n);
            pair<Dataframe, Dataframe> split_train_test(double train_percent);
            void extend(Dataframe other);

            Dataframe random();
    };

    Dataframe::Dataframe()
    {
    }

    Dataframe read_pat(const string& filename){
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Could not open file");
        }

        string line;

        vector<vector<double>> table;
        vector<double> row;

        bool isFirst = true;
        while (getline(file, line)) {
            if(line[0] == 'p'){
                if(isFirst){
                    isFirst = false;
                    continue;
                }
                table.push_back(row);
                row.clear();
            }else{
                regex re("\\s+");
                line = regex_replace(line, re, " ");
                stringstream ss(line);
                string value;
                while (getline(ss, value, ' ')) {
                    row.push_back(stod(value));
                }
            }
        }

        return Dataframe(table);
    }

    Dataframe read_csv(const string& filename, int start = 0){
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Could not open file");
        }

        string line;

        vector<vector<double>> table;
        vector<double> row;
        
        int count = 0;
        while (getline(file, line)) {
            if(count >= start){
                stringstream ss(line);
                string value;
                while (getline(ss, value, ',')) {
                    row.push_back(stod(value));
                }
                table.push_back(row);
                row.clear();
            }
            count++;
        }
        return Dataframe(table);
    }

    Dataframe Dataframe::get_onehot(const int& column) const {
        if (column < 0 || column >= this->table[0].size()) {
            throw std::out_of_range("Invalid column index");
        }

        unordered_map<double, int> unique_values;
        int next_index = 0;

        for (const auto& row : this->table) {
            double value = row[column];
            if (unique_values.find(value) == unique_values.end()) {
                unique_values[value] = next_index++;
            }
        }

        std::vector<vector<double>> onehot_table;

        for (const auto& row : this->table) {
            vector<double> onehot_row(next_index, 0.0);
            double value = row[column];
            onehot_row[unique_values[value]] = 1.0;

            onehot_table.push_back(onehot_row);
        }

        return Dataframe(onehot_table);
    }

    Dataframe::Dataframe(const string& file_path, int start_line){
        ifstream file(file_path);
        if (!file.is_open()) {
            throw runtime_error("Could not open file");
        }

        string line;
        int line_count = 0;

        vector<vector<double>> table;

        while (getline(file, line)) {
            line_count++;
            if(line_count >= start_line){
                vector<double> row;
                stringstream ls(line);
                string data;
                int count = 0;
                while (getline(ls, data, '\t')) {
                    row.push_back(stod(data));
                    count++;
                }
                width = count;
                table.push_back(row);
            }
        }
        file.close();
        this->table = table;
    }

    Dataframe::Dataframe(vector<vector<double>> table){
        this->table = table;
        if (!table.empty()) {
            width = table[0].size();
        } else {
            width = 0;
        }
    }

    Dataframe::~Dataframe()
    {
    }

    void Dataframe::to_csv(string filename){
        ofstream myfile;
        myfile.open (filename);
        for(int r = 0; r < this->get_depth(); r++){
            for(int c = 0; c < this->get_width(); c++){
                if(c == 0) myfile << this->get(r, c);
                else myfile << "," << this->get(r, c);
            }
            myfile << "\n";
        }
        myfile.close();
    }

    int Dataframe::get_width() const{
        return width;
    }

    int Dataframe::get_depth() const{
        return table.size();
    }

    double Dataframe::get(int r, int c) const{
        if (r >= get_depth() || c >= get_width()) {
            throw out_of_range("Index out of range");
        }
        return table[r][c];
    }

    vector<double> Dataframe::getRow(int r) const{
        return table[r];
    }

    void Dataframe::insert(vector<double> row){
        if(this->width != 0){
            if(this->width != row.size()){
                runtime_error("Dataframe width not match");
            }
        }else{
            this->width = row.size();
        }
        this->table.push_back(row);
    }

    Dataframe Dataframe::get_column(const vector<int>& columns) const{
        vector<vector<double>> new_table;
        for(int r = 0; r < this->get_depth(); r++){
            vector<double> row;
            for(int c : columns){
                row.push_back(this->get(r, c));
            }
            new_table.push_back(row);
        }
        return Dataframe(new_table);
    }

    Dataframe Dataframe::get_column_without(const vector<int>& columns) const{
        vector<int> selected;
        for(int i = 0; i < this->width; i++){
            bool is_needed = true;
            for(int c : columns){
                if(i == c){
                    is_needed = false;
                    break;
                }
            }
            if(is_needed) selected.push_back(i);
        }
        return get_column(selected);
    }

    Dataframe Dataframe::random(){
        srand(time(0));

        vector<vector<double>> random_table;
        while(!table.empty()){
            int range = (table.size() - 1) + 1;
            int num = rand() % range;

            random_table.push_back(table[num]);
            table.erase(table.begin() + num);
        }
        return Dataframe(random_table);
    }

    vector<Dataframe> Dataframe::split_n(int n){
        vector<vector<vector<double>>> table_ls(n);
        for(int i = 0; i < this->get_depth(); i++){
            table_ls[i%n].push_back(this->getRow(i));
        }

        vector<Dataframe> df_ls;
        for(vector<vector<double>> table : table_ls){
            df_ls.push_back(Dataframe(table));
        }
        return df_ls;
    }

    pair<Dataframe, Dataframe> Dataframe::split_train_test(double train_percent) {
        if (train_percent < 0.0 || train_percent > 1.0) {
            throw invalid_argument("Training percentage must be between 0.0 and 1.0");
        }

        // Shuffle the data to ensure random splitting
        srand(static_cast<unsigned int>(time(0)));
        vector<vector<double>> shuffled_table = this->table;
        random_shuffle(shuffled_table.begin(), shuffled_table.end());

        int train_size = static_cast<int>(train_percent * shuffled_table.size());

        // Create train and test tables
        vector<vector<double>> train_table(shuffled_table.begin(), shuffled_table.begin() + train_size);
        vector<vector<double>> test_table(shuffled_table.begin() + train_size, shuffled_table.end());

        // Return as a pair of Dataframes
        return { Dataframe(train_table), Dataframe(test_table) };
    }

    void Dataframe::extend(Dataframe other){
        if(this->width != 0){
            if(this->width != other.get_width()){
                runtime_error("Dataframe width not match");
            }
        }else{
            this->width = other.get_width();
        }
        for(int i = 0; i < other.get_depth(); i++){
            this->table.push_back(other.getRow(i));
        }
    }

    Dataframe merge(vector<Dataframe> df_ls){
        Dataframe df_merged;
        for(Dataframe df : df_ls){
            df_merged.extend(df);
        }
        return df_merged;
    }
}


#endif
