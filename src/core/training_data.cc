#include "core/training_data.h"

namespace naivebayes {
    TrainingData::TrainingData(const string& labels_path) {
        ParseImageLabels(labels_path);

        //initialize training_data_images_ so it is indexable
        for (size_t i = 0; i <= kEndingLabelValue; ++i) {
            training_data_images_.emplace_back(vector<Image>());
        }
    }

    void TrainingData::ParseImageLabels(const string& labels_path) {
        ifstream infile(labels_path); string line;
        while (getline(infile, line)) {
            image_labels_.emplace_back(std::stoi(line));
        }
    }

    ifstream& operator >> (ifstream& in, TrainingData& training_data) {
        size_t image_height = training_data.GetImageHeightFromFile(in);

        string pixels_row;
        //iterating through each image in images file
        for (size_t i = 0; i < training_data.image_labels_.size(); ++i) {
            Image image;
            for (size_t j = 0; j < image_height; ++j) {
                getline(in, pixels_row);
                image.AddPixelsRow(pixels_row);
            }
            //add image to list of images at corresponding index and reset row count
            training_data.training_data_images_[training_data.image_labels_[i]].push_back(image);
        }
        return in;
    }

    size_t TrainingData::GetImageHeightFromFile(ifstream& in) {
        //length of first line of file = height of each image
        string firstLine;
        getline(in, firstLine);
        size_t image_height = firstLine.length();
        in.seekg(0, std::ios_base::beg); //move stream back to first line
        
        image_height_ = image_height;
        return image_height;
    }

    /* Getters */
    const vector<vector<Image>>& TrainingData::GetTrainingDataImages() const { return training_data_images_; }
    size_t TrainingData::GetImageHeight() const { return image_height_; }
    int TrainingData::GetTotalNumberOfImages() const { return image_labels_.size(); }
}