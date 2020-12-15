#pragma once

#include "core/image.h"
#include <fstream>

using std::ifstream;

namespace naivebayes {
    class TrainingData {
        public:
            /* Parameterized constructor: initializes labels_path vector and training_data_images_ */
            explicit TrainingData(const string& labels_path);
    
            /* Overloads >> operator; reads in file of images and stores in training_data_images_ */
            friend ifstream& operator >> (ifstream& in, TrainingData& training_data);
    
            /* Getters */
            const vector<vector<Image>>& GetTrainingDataImages() const;
            size_t GetImageHeight() const;
            int GetTotalNumberOfImages() const;
    
        private:
            vector<vector<Image>> training_data_images_; //list of images for each class
            vector<int> image_labels_;
            size_t image_height_;
            const size_t kEndingLabelValue = 9; //labels go from 0 to kEndingLabelValue
    
            /* Stores labels in labels_path into image_labels_ vector */
            void ParseImageLabels(const string& labels_path);
    
            /* Gets the image height from the file of images and stores in image_height_ */
            size_t GetImageHeightFromFile(ifstream& in);
    };
}