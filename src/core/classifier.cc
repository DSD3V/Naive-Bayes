#include "core/classifier.h"

namespace naivebayes {
    Classifier::Classifier(TrainingModel& training_model) :
    training_model_(&training_model) {}

    int Classifier::Classify(const Image& image) {
        SetLikelihoodScores(image);

        double max = -DBL_MAX; int max_index;
        for (size_t i = 0; i < likelihood_scores_.size(); ++i) {
            if (likelihood_scores_[i] > max) {
                max = likelihood_scores_[i];
                max_index = i;
            }
        }
        
        return max_index; //this is the classification; index of the highest likelihood score
    }

    double Classifier::TestAccuracy(const string& test_images_path, const string& test_labels_path) {
        vector<pair<int, int>> classifications; //first is actual label, second is computed label
        
        //store actual labels as first of element of each pair in classifications
        ifstream in_test_labels(test_labels_path); string line;
        while (getline(in_test_labels, line)) {
            classifications.emplace_back(std::make_pair(std::stoi(line), 0));
        }

        //get height of each image
        ifstream in_test_images(test_images_path); string first_line;
        getline(in_test_images, first_line);
        size_t image_height = first_line.length();
        in_test_images.seekg(0, std::ios_base::beg); //move stream back to first line

        //iterate through test_images_path file, constructing and classifying each image
        string pixels_row;
        for (auto& classification : classifications) {
            Image image;
            for (size_t j = 0; j < image_height; ++j) {
                getline(in_test_images, pixels_row);
                image.AddPixelsRow(pixels_row);
            }
            //reached end of image: classify image
            classification.second = Classify(image);
        }
        
        //Check and return accuracy of classifications
        return ComputeAccuracy(classifications);
    }

    void Classifier::SetLikelihoodScores(const Image& image) {
        likelihood_scores_.clear();
        
        //first add class probabilities
        for (const double prob : training_model_->GetClassProbs()) {
            likelihood_scores_.emplace_back(log(prob));
        }
        
        //now compute likelihood scores
        for (size_t y = 0; y < image.image_pixels_.size(); ++y) {
            for (size_t x = 0; x < image.image_pixels_[y].size(); ++x) {
                for (size_t i = 0; i < training_model_->GetShadedProbs().size(); ++i) {
                    //if the pixel is shaded, add the probability that it is shaded, else 1 minus it
                    likelihood_scores_[i] += (image.image_pixels_[y][x] == 1) ? log(training_model_->GetShadedProbs()[i][y][x])
                                                                           : log(1 - training_model_->GetShadedProbs()[i][y][x]);
                }
            }
        }
    }

    double Classifier::ComputeAccuracy(const vector<pair<int, int>>& classifications) const {
        int correct = 0, total = 0;
        for (const auto& classification : classifications) {
            if (classification.first == classification.second) correct++;
            total++;
        }
        double accuracy = (correct / (double) total) * 100;

        return accuracy;
    }

    const vector<double>& Classifier::GetLikelihoodScores() const { return likelihood_scores_; }
}