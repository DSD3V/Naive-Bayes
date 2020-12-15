#pragma once

#include "core/training_model.h"
#include "image.h"
#include <utility>

using std::pair;

namespace naivebayes {
    class Classifier {
        public:
            /* Default constructor */
            Classifier() {};
        
            /* Parameterized constructor: initializes Classifier with the given training model */
            Classifier(TrainingModel& training_model);
            
            /* Classifies the given image; returns the computed class label */
            int Classify(const Image& image);
            
            /* Tests and returns the accuracy of the classifier by classifying the images in test_images_path */
            double TestAccuracy(const string& test_images_path, const string& test_labels_path);
            
            /* Getter for likelihood scores, used for testing accuracy of calculations */
            const vector<double>& GetLikelihoodScores() const;
    
        private:
            TrainingModel* training_model_; //Model used for classifying
            
            //Likelihood scores of the last image that was classified; used to test that calculations are correct
            vector<double> likelihood_scores_;
            
            /* Helper methods for classifying an image */
            void SetLikelihoodScores(const Image& image);
            double ComputeAccuracy(const vector<pair<int, int>>& classifications) const; //if first == second, classification is correct
    };
}