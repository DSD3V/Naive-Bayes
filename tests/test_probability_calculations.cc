#include "core/training_data.h"
#include "core/training_model.h"
#include <catch2/catch.hpp>
#include <limits>

namespace naivebayes {
    const string training_images = "tests_data/test_trainingimages.txt";
    const string bigger_images = "tests_data/test_trainingimages_bigger.txt";
    const string training_labels = "tests_data/test_traininglabels.txt";

    TEST_CASE("Test that probabilities are stored correctly") {
        ifstream in(training_images);
        TrainingData training_data(training_labels);
        in >> training_data;
        TrainingModel training_model(training_data);

        SECTION("Test that class probabilities are calculated correctly") {
            //create expected output
            vector<double> expected_class_probs;
            expected_class_probs.push_back((1.0 + 0.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 4.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 3.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 3.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 0.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 0.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 0.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 0.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 0.0) / (10.0 + 10.0));
            expected_class_probs.push_back((1.0 + 0.0) / (10.0 + 10.0));
        
            //get actual output
            vector<double> actual_class_probs = training_model.GetClassProbs();
        
            REQUIRE(actual_class_probs[0] == Approx(expected_class_probs[0]).epsilon(0.05));
            REQUIRE(actual_class_probs[1] == Approx(expected_class_probs[1]).epsilon(0.05));
            REQUIRE(actual_class_probs[2] == Approx(expected_class_probs[2]).epsilon(0.05));
            REQUIRE(actual_class_probs[3] == Approx(expected_class_probs[3]).epsilon(0.05));
            REQUIRE(actual_class_probs[4] == Approx(expected_class_probs[4]).epsilon(0.05));
            REQUIRE(actual_class_probs[9] == Approx(expected_class_probs[9]).epsilon(0.05));
        }
    
        SECTION("Test that calculate method calculates shaded probabilities correctly") {
            //create expected output
            vector<vector<vector<double>>> expected_probs;
            
            vector<vector<double>> v0 = {
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0}
            };
            
            vector<vector<double>> v1 = {
                    {1 / 6.0, 1 / 3.0, 5 / 6.0, 1 / 3.0, 1 / 6.0, 1 / 6.0},
                    {1 / 6.0, 2 / 3.0, 5 / 6.0, 1 / 3.0, 1 / 6.0, 1 / 6.0},
                    {1 / 6.0, 1 / 2.0, 5 / 6.0, 1 / 3.0, 1 / 6.0, 1 / 6.0},
                    {1 / 6.0, 1 / 3.0, 5 / 6.0, 1 / 3.0, 1 / 6.0, 1 / 6.0},
                    {1 / 2.0, 2 / 3.0, 5 / 6.0, 2 / 3.0, 1 / 2.0, 1 / 6.0},
                    {1 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 6.0}
            };
            
            vector<vector<double>> v2 = {
                    {4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 2 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 2 / 5.0, 4 / 5.0, 2 / 5.0},
                    {2 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 3 / 5.0, 2 / 5.0},
                    {4 / 5.0, 2 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0},
                    {3 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 2 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0}
            };
            
            vector<vector<double>> v3 = {
                    {4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 3 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 3 / 5.0, 3 / 5.0, 3 / 5.0, 1 / 5.0},
                    {4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 3 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 3 / 5.0, 3 / 5.0, 3 / 5.0, 1 / 5.0},
                    {4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 2 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0}
            };
            
            vector<vector<double>> v9 = {
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0}
            };
            
            expected_probs.push_back(v0);
            expected_probs.push_back(v1);
            expected_probs.push_back(v2);
            expected_probs.push_back(v3);
            expected_probs.push_back(v9);
            
            //get actual output
            vector<vector<vector<double>>> actual_probs = training_model.GetShadedProbs();
            
            REQUIRE(actual_probs[0] == expected_probs[0]);
            REQUIRE(actual_probs[1] == expected_probs[1]);
            REQUIRE(actual_probs[2] == expected_probs[2]);
            REQUIRE(actual_probs[3] == expected_probs[3]);
            REQUIRE(actual_probs[9] == expected_probs[4]);
        }
    }

    TEST_CASE("Test that probabilities are stored correctly for bigger images") {
        ifstream in(bigger_images);
        TrainingData training_data(training_labels);
        in >> training_data;
        TrainingModel training_model(training_data);
    
        SECTION("Test that calculate method calculates class probabilities correctly for bigger images") {
            //create expected output
            vector<double> expected_class_probs;
            expected_class_probs.push_back(1.0 / 20.0);
            expected_class_probs.push_back(5.0 / 20.0);
            expected_class_probs.push_back(4.0 / 20.0);
            expected_class_probs.push_back(4.0 / 20.0);
            expected_class_probs.push_back(1.0 / 20.0);
            
            //get actual output
            vector<double> actual_class_probs = training_model.GetClassProbs();
            
            REQUIRE(actual_class_probs[0] == Approx(expected_class_probs[0]).epsilon(0.05));
            REQUIRE(actual_class_probs[1] == Approx(expected_class_probs[1]).epsilon(0.05));
            REQUIRE(actual_class_probs[2] == Approx(expected_class_probs[2]).epsilon(0.05));
            REQUIRE(actual_class_probs[3] == Approx(expected_class_probs[3]).epsilon(0.05));
            REQUIRE(actual_class_probs[9] == Approx(expected_class_probs[4]).epsilon(0.05));
            }
            
            SECTION("Test that calculate method calculates shaded probabilities correctly for bigger images") {
            //create expected output
            vector<vector<vector<double>>> expected_probs;
            
            vector<vector<double>> v3 = {
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 2 / 5.0, 3 / 5.0, 4 / 5.0, 2 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 2 / 5.0, 3 / 5.0, 4 / 5.0, 2 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 2 / 5.0, 3 / 5.0, 4 / 5.0, 2 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 2 / 5.0, 3 / 5.0, 4 / 5.0, 2 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 2 / 5.0, 3 / 5.0, 4 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 4 / 5.0, 3 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0},
                    {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0}
            };
            
            vector<vector<double>> v5 = {
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0},
                    {1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0, 1 / 2.0}
            };
            
            expected_probs.push_back(v3);
            expected_probs.push_back(v5);
            
            //get actual output
            vector<vector<vector<double>>> actual_probs = training_model.GetShadedProbs();
            
            REQUIRE(actual_probs[3] == expected_probs[0]);
            REQUIRE(actual_probs[5] == expected_probs[1]);
        }
    }
}