#include "core/training_data.h"
#include "core/training_model.h"
#include "core/classifier.h"
#include <catch2/catch.hpp>
#include <limits>

namespace naivebayes {
    const string training_images = "tests_data/test_trainingimages.txt";
    const string training_labels = "tests_data/test_traininglabels.txt";
    const string model_file = "tests_data/test_output_file.txt";

    TEST_CASE("Test that classifier works correctly") {
        SECTION("Test that classifier correctly calculates likelihood scores") {
            //create expected output
            vector<vector<int>> image1_pixels {
                    {0, 1, 1, 1, 0, 0},
                    {0, 1, 1, 1, 0, 0},
                    {0, 1, 1, 1, 0, 0},
                    {0, 1, 1, 1, 0, 0},
                    {0, 1, 1, 1, 0, 0},
                    {0, 0, 0, 0, 0, 0}
            };
            Image image1(image1_pixels);
        
            vector<double> expected_likelihoodscores;
        
            double expected_likelihoodscore_0 =
                    log(0.05) +
                    log(1 - 0.5) + log(0.5) + log(0.5) + log(0.5) + log(1 - 0.5) + log(1 - 0.5) +
                    log(1 - 0.5) + log(0.5) + log(0.5) + log(0.5) + log(1 - 0.5) + log(1 - 0.5) +
                    log(1 - 0.5) + log(0.5) + log(0.5) + log(0.5) + log(1 - 0.5) + log(1 - 0.5) +
                    log(1 - 0.5) + log(0.5) + log(0.5) + log(0.5) + log(1 - 0.5) + log(1 - 0.5) +
                    log(1 - 0.5) + log(0.5) + log(0.5) + log(0.5) + log(1 - 0.5) + log(1 - 0.5) +
                    log(1 - 0.5) + log(1 - 0.5) + log(1 - 0.5) + log(1 - 0.5) + log(1 - 0.5) + log(1 - 0.5);
        
            double expected_likelihoodscore_1 =
                    log(0.25) +
                    log(1 - 0.166667) + log(0.333333) + log(0.833333) + log(0.333333) + log(1 - 0.166667) + log(1 - 0.166667) +
                    log(1 - 0.166667) + log(0.666667) + log(0.833333) + log(0.333333) + log(1 - 0.166667) + log(1 - 0.166667) +
                    log(1 - 0.166667) + log(0.5) + log(0.833333) + log(0.333333) + log(1 - 0.166667) + log(1 - 0.166667) +
                    log(1 - 0.166667) + log(0.333333) + log(0.833333) + log(0.333333) + log(1 - 0.166667) + log(1 - 0.166667) +
                    log(1 - 0.5) + log(0.666667) + log(0.833333) + log(0.666667) + log(1 - 0.5) + log(1 - 0.166667) +
                    log(1 - 0.166667) + log(1 - 0.166667) + log(1 - 0.166667) + log(1 - 0.166667) + log(1 - 0.166667) + log(1 - 0.166667);
        
            double expected_likelihoodscore_3 =
                    log(0.2) +
                    log(1 - 0.8) + log(0.8) + log(0.8) + log(0.8) + log(1 - 0.6) + log(1 - 0.2) +
                    log(1 - 0.2) + log(0.2) + log(0.6) + log(0.6) + log(1 - 0.6) + log(1- 0.2) +
                    log(1 - 0.8) + log(0.8) + log(0.8) + log(0.8) + log(1 - 0.6) + log(1 - 0.2) +
                    log(1 - 0.2) + log(0.2) + log(0.6) + log(0.6) + log(1 - 0.6) + log(1 - 0.2) +
                    log(1 - 0.8) + log(0.8) + log(0.8) + log(0.8) + log(1 - 0.4) + log(1 - 0.2) +
                    log(1 - 0.2) + log(1 - 0.2) + log(1 - 0.2) + log(1 - 0.2) + log(1 - 0.2) + log(1 - 0.2);
            
            expected_likelihoodscores.push_back(expected_likelihoodscore_0);
            expected_likelihoodscores.push_back(expected_likelihoodscore_1);
            expected_likelihoodscores.push_back(expected_likelihoodscore_3);
        
            //get actual output
            ifstream in(model_file);
            TrainingModel training_model;
            in >> training_model;
            
            Classifier classifier(training_model);
            
            classifier.Classify(image1);
        
            REQUIRE(classifier.GetLikelihoodScores()[0] == Approx(expected_likelihoodscores[0]).epsilon(0.005));
            REQUIRE(classifier.GetLikelihoodScores()[1] == Approx(expected_likelihoodscores[1]).epsilon(0.005));
            REQUIRE(classifier.GetLikelihoodScores()[3] == Approx(expected_likelihoodscores[2]).epsilon(0.005));
        }
    
        SECTION("Test that classifier correctly classifies test images dataset") {
            //read training_model from file into empty training model
            TrainingModel training_model;
            ifstream in_trainingmodel("../data/training_model.txt");
            in_trainingmodel >> training_model;

            Classifier classifier(training_model);

            //classify each image and determine accuracy using test images and labels
            double accuracy = classifier.TestAccuracy("../data/test_images.txt", "../data/test_labels.txt");

            REQUIRE(accuracy >= 75.0);
        }
    }
}