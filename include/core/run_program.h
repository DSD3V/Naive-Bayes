#pragma once

#include "core/training_data.h"
#include "core/training_model.h"
#include "core/classifier.h"
#include <iostream>

using std::cout;
using std::endl;

namespace naivebayes {
    /* Trains the model with given images and stores in output_model_file */
    void TrainModel(const string& training_images, const string& training_labels, const string& output_model_file) {
        //create TrainingData object
        TrainingData training_data(training_labels);
        ifstream in_data(training_images);
        if (in_data.fail()) {
            cout << "Failed to load training images file. Program terminating.";
            return;
        }
        
        cout << "Training model with images stored at " << training_images << "." << endl;
        
        in_data >> training_data;

        //create TrainingModel object
        TrainingModel training_model(training_data);
        ofstream out(output_model_file);

        //store model in output file
        out << training_model;

        cout << "Model successfully written to " << output_model_file << "." << endl;
    }

    /* Loads a model from model_file into a TrainingModel object */
    void LoadModel(const string& model_file) {
        //create empty TrainingModel
        TrainingModel training_model;

        //load into TrainingModel
        ifstream in_model(model_file);
        if (in_model.fail()) {
            std::cout << "Failed to load model file. Program terminating.";
            return;
        }
        
        cout << "Loading model stored at " << model_file << "." << endl;
        
        in_model >> training_model;
        
        cout << "Model successfully loaded into TrainingModel object." << endl;
    }

    /* Loads a model from model_file into a TrainingModel and tests its classification accuracy */
    void LoadAndTest(const string& model_file, const string& test_images, const string& test_labels) {
        //create empty TrainingModel
        TrainingModel training_model;

        //load into TrainingModel
        ifstream in_model(model_file);
        if (in_model.fail()) {
            std::cout << "Failed to load model file. Program terminating.";
            return;
        }

        cout << "Loading model stored at " << model_file << "." << endl;
        
        in_model >> training_model;

        cout << "Model successfully loaded into TrainingModel object." << endl;
        
        Classifier classifier(training_model);
        
        //test model, print accuracy
        cout << "Classification accuracy using model: " << classifier.TestAccuracy(test_images, test_labels) << endl;
    }

    /* Trains a model, stores it in output_model_file, and then tests its classification accuracy */
    void TrainAndTest(const string& training_images, const string& training_labels, const string& output_model_file,
                      const string& test_images, const string& test_labels) {
        //create TrainingData object
        TrainingData training_data(training_labels);
        ifstream in_data(training_images);
        if (in_data.fail()) {
            std::cout << "Failed to load training images file. Program terminating.";
            return;
        }

        cout << "Training model with images stored at " << training_images << "." << endl;
        
        in_data >> training_data;

        //create and store TrainingModel
        TrainingModel training_model(training_data);
        ofstream out_model(output_model_file);
        out_model << training_model;

        cout << "Model successfully trained and written to " << output_model_file << "." << endl;
        
        Classifier classifier(training_model);

        //test model and print accuracy
        cout << "Classification accuracy using model: " << classifier.TestAccuracy(test_images, test_labels) << "%" << endl;
    }
}