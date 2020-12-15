#include "core/training_model.h"
#include <sstream>

namespace naivebayes {
    TrainingModel::TrainingModel(TrainingData& training_data) {
        image_height_ = training_data.GetImageHeight();
        CalculateProbs(training_data);
    }

    /* Methods for calculating probabilities */

    void TrainingModel::CalculateProbs(TrainingData& training_data) {
        //first, computes class probabilities and stores in class_probs_ vector
        ComputeClassProbs(training_data);

        //next, computes pixel probabilities for each class of images and store result in shaded_probs_
        for (const auto& images : training_data.GetTrainingDataImages()) {
            ComputePixelProbs(images);
        }
    }

    void TrainingModel::ComputePixelProbs(const vector<Image>& images) {
        //initialize 2d vector of shaded probabilities for this class of images
        vector<vector<double>> shaded_probs(image_height_, vector<double>(image_height_));

        //initialize 2d vector which will track number of times each pixel is shaded to 0s
        vector<vector<int>> num_times_shaded(image_height_, vector<int>(image_height_, 0));

        for (const auto& image : images) {
            for (size_t y = 0; y < image.image_pixels_.size(); ++y) {
                for (size_t x = 0; x < image.image_pixels_[y].size(); ++x) {
                    if (image.image_pixels_[y][x] == 1) num_times_shaded[y][x]++;
                }
            }
        }

        //now iterate through shaded_probs and fill with calculated probabilities
        double numerator, denominator = (kNumPixelClasses * kLaplaceK) + images.size();
        for (size_t x = 0; x < image_height_; ++x) {
            for (size_t y = 0; y < image_height_; ++y) {
                numerator = kLaplaceK + num_times_shaded[x][y];
                shaded_probs[x][y] = numerator / denominator;
            }
        }
        shaded_probs_.push_back(shaded_probs);
    }

    void TrainingModel::ComputeClassProbs(TrainingData& training_data) {
        int total_num_of_images = training_data.GetTotalNumberOfImages();
        int total_num_of_classes = training_data.GetTrainingDataImages().size();

        double numerator, denominator;
        //iterating through training_data list of images
        for (const auto& images : training_data.GetTrainingDataImages()) {
            numerator = kLaplaceK + images.size();
            denominator = (total_num_of_classes * kLaplaceK) + total_num_of_images;
            class_probs_.push_back(numerator / denominator);
        }
    }
    
    /* Methods for reading in model from a file */

    ifstream& operator >> (ifstream& in, TrainingModel& training_model) {
        string line;
        //store image height and number of classes
        getline(in, line);
        training_model.image_height_ = std::stoi(line);
        getline(in, line);
        int num_of_classes = std::stoi(line);

        //store class probabilities
        training_model.ReadClassProbs(in, training_model, num_of_classes);

        //store shaded probabilities
        training_model.ReadShadedProbs(in, training_model);

        return in;
    }

    void TrainingModel::ReadClassProbs(ifstream& in, TrainingModel& training_model, size_t num_of_classes) {
        string line;
        vector<double> class_probs;
        for (size_t i = 0; i < num_of_classes; ++i) {
            getline(in, line);
            class_probs_.emplace_back(std::stod(line));
        }
    }

    void TrainingModel::ReadShadedProbs(ifstream& in, TrainingModel& training_model) {
        string line, prob;
        vector<double> row;
        vector<vector<double>> shaded_probs;
        size_t line_count = 0;

        for (size_t i = 0; i < class_probs_.size(); ++i) {
            //iterate through 2d vector of probabilities for each class
            for (size_t j = 0; j < image_height_; ++j) {
                getline(in, line);
                std::istringstream stream(line);
                //iterate through line of file, storing double values into row
                while (getline(stream, prob, ' ') && line_count < image_height_) {
                    row.emplace_back(std::stod(prob));
                    line_count++;
                }
                //reached end of line, add row to shaded_probs and reset
                shaded_probs.push_back(row);
                line_count = 0;
                row.clear();
            }
            //reached end of probs for this class, add to shaded_probs_ and reset
            shaded_probs_.push_back(shaded_probs);
            shaded_probs.clear();
        }
    }

    /* Methods for writing model to a file */

    ofstream& operator << (ofstream& out, TrainingModel& training_model) {
        //write image height and number of classes to file
        out << training_model.image_height_ << '\n';
        out << training_model.class_probs_.size() << '\n';

        //write class probabilities to file
        training_model.WriteClassProbs(out, training_model);

        //write shaded probabilities to file
        training_model.WriteShadedProbs(out, training_model);

        return out;
    }

    void TrainingModel::WriteClassProbs(ofstream& out, TrainingModel& training_model) const {
        for (const double probability : class_probs_) {
            out << probability << '\n';
        }
    }

    void TrainingModel::WriteShadedProbs(ofstream& out, TrainingModel& training_model) const {
        for (const auto& probs : shaded_probs_) {
            for (const auto& row : probs) {
                for (const double prob : row) {
                    out << prob << ' ';
                }
                out << '\n';
            }
        }
    }

    /* Getters */
    const vector<vector<vector<double>>>& TrainingModel::GetShadedProbs() const { return shaded_probs_; }
    const vector<double>& TrainingModel::GetClassProbs() const { return class_probs_; }
}