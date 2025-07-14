#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

void readCSV(const string& filename, vector<vector<double> >& features, vector<int>& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        cout << "Raw line: " << line << endl;

        stringstream ss(line);
        vector<double> row;
        double value;
        int label;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',' || ss.peek() == ' ') {
                ss.ignore();
            }
        }

        cout << "Processed row: ";
        for (size_t i = 0; i < row.size(); ++i) {
            double v = row[i];
            cout << v << " ";
        }
        cout << endl;

        if (!row.empty()) {
            label = static_cast<int>(row.back());
            row.pop_back();
        } else {
            cerr << "Error: Empty row encountered." << endl;
            continue;
        }

        features.push_back(row);
        labels.push_back(label);
    }
    
    file.close();
}

void trainPerceptron(vector<vector<double> >& features, vector<int>& labels, 
                    vector<double>& weights, double& bias, double learningRate, int epochs) {
    int nSamples = features.size();
    int nFeatures = features[0].size();

    for (int epoch = 0; epoch < epochs; epoch++) {
        int errorCount = 0;
        
        for (int i = 0; i < nSamples; i++) {
            double weightedSum = bias;

            for (int j = 0; j < nFeatures; j++) {
                weightedSum += weights[j] * features[i][j];
            }

            int prediction = (weightedSum >= 0) ? 1 : -1;
            int error = labels[i] - prediction;

            if (error != 0) {
                errorCount++;
                for (int j = 0; j < nFeatures; j++) {
                    weights[j] += learningRate * error * features[i][j];
                }
                bias += learningRate * error;
            }
        }

        cout << "Epoch " << epoch + 1 << ": " << errorCount << " errors" << endl;

        if (errorCount == 0) {
            cout << "Perceptron is fully trained after " << epoch + 1 << " epochs." << endl;
            break;
        }
    }
}

vector<int> predictPerceptron(const vector<vector<double> >& features, 
                             const vector<double>& weights, double bias) {
    vector<int> predictions;
    
    for (size_t i = 0; i < features.size(); ++i) {
        double weightedSum = bias;
        
        for (size_t j = 0; j < features[i].size(); ++j) {
            weightedSum += weights[j] * features[i][j];
        }
        
        int prediction = (weightedSum >= 0) ? 1 : -1;
        predictions.push_back(prediction);
    }
    
    return predictions;
}

double calculateAccuracy(const vector<int>& trueLabels, const vector<int>& predictedLabels) {
    int correct = 0;
    
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        if (trueLabels[i] == predictedLabels[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / trueLabels.size();
}

int main() {
    string trainFilename = "training_data.csv";
    string testFilename = "test_data.csv";
    vector<vector<double> > trainFeatures, testFeatures;
    vector<int> trainLabels, testLabels;

    cout << "Reading training CSV file..." << endl;
    readCSV(trainFilename, trainFeatures, trainLabels);

    cout << "Reading test CSV file..." << endl;
    readCSV(testFilename, testFeatures, testLabels);

    if (trainFeatures.empty() || trainLabels.empty()) {
        cerr << "Error: No data loaded from training CSV file!" << endl;
        return -1;
    }
    
    if (testFeatures.empty() || testLabels.empty()) {
        cerr << "Error: No data loaded from test CSV file!" << endl;
        return -1;
    }
    
    if (trainFeatures.size() != trainLabels.size() || testFeatures.size() != testLabels.size()) {
        cerr << "Error: Features and labels size mismatch!" << endl;
        return -1;
    }

    int nFeatures = trainFeatures[0].size();
    vector<double> weights(nFeatures);
    weights[0] = 1.0;
    weights[1] = 1.0;
    
    if (nFeatures > 2) {
        for (int i = 2; i < nFeatures; ++i) {
            weights[i] = 0.0;
        }
    }
    
    double bias = 1.0;
    double learningRate = 1.0;
    int epochs = 1000;

    cout << "Training perceptron..." << endl;
    trainPerceptron(trainFeatures, trainLabels, weights, bias, learningRate, epochs);

    cout << "Trained Weights:" << endl;
    for (size_t i = 0; i < weights.size(); i++) {
        cout << "Weight[" << i << "] = " << weights[i] << endl;
    }
    cout << "Bias: " << bias << endl;

    cout << "Predicting on test data..." << endl;
    vector<int> predictions = predictPerceptron(testFeatures, weights, bias);

    double accuracy = calculateAccuracy(testLabels, predictions);
    cout << "Test Accuracy: " << accuracy * 100 << "%" << endl;

    return 0;
}