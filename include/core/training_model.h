#pragma once

#include "core/image.h"
#include "core/training_data.h"

using std::ofstream;

namespace naivebayes {
    class TrainingModel {
        public:
            /* Default constructor; used when loading in a training model from file */
            TrainingModel() = default;
            
            /* Parameterized constructor: trains and stores the model using the given training_data */
            explicit TrainingModel(TrainingData& training_data);
    
            /* Overloads << operator: writes probabilities to a file */
            friend ofstream& operator << (ofstream& out, TrainingModel& training_model);
            
            /* Overloads >> operator: reads probabilities from a file into model */
            friend ifstream& operator >> (ifstream& in, TrainingModel& training_model);
    
            /* Getters */
            const vector<vector<vector<double>>>& GetShadedProbs() const;
            const vector<double>& GetClassProbs() const;
    
        private:
            const int kLaplaceK = 1; //laplace constant for calculating probabilities
            const int kNumPixelClasses = 2; //pixels are either shaded or unshaded
            vector<vector<vector<double>>> shaded_probs_; //stores probabilities for each class
            vector<double> class_probs_; //stores class probabilities
            size_t image_height_ = 0;
    
            /* Computes class and pixel probabilities based on training_data */
            void CalculateProbs(TrainingData& training_data);
            
            /* Computes class probabilities and stores in class_probs_ */
            void ComputeClassProbs(TrainingData& training_data);
            
            /* Computes pixel probabilities and stores in shaded_probs_ */
            void ComputePixelProbs(const vector<Image>& images);
    
            /* Helper methods for reading in from file */
            void ReadClassProbs(ifstream& in, TrainingModel& training_model, size_t num_of_classes);
            void ReadShadedProbs(ifstream& in, TrainingModel& training_model);
    
            /* Helper methods for writing to file */
            void WriteClassProbs(ofstream& out, TrainingModel& training_model) const;
            void WriteShadedProbs(ofstream& out, TrainingModel& training_model) const;
    };
}