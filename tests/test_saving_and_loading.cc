#include "core/training_data.h"
#include "core/training_model.h"
#include <catch2/catch.hpp>
#include <limits>

namespace naivebayes {
    const string training_images = "tests_data/test_trainingimages.txt";
    const string bigger_images = "tests_data/test_trainingimages_bigger.txt";
    const string training_labels = "tests_data/test_traininglabels.txt";
    const string output_file = "tests_data/test_output_file.txt";
    const string output_file2 = "tests_data/test_output_file2.txt";

    TEST_CASE("Test that probabilities are written to file correctly") {
        TrainingData training_data(training_labels);
        SECTION("Test that << operator correctly writes probability data to file") {
            //write to file
            ifstream in(training_images);
            in >> training_data;
            TrainingModel training_model(training_data);
            ofstream out(output_file);
            out << training_model;
            out.close();
        
            //read from file that was written to; check beginning, middle, and end
            ifstream actual(output_file); string s;
            int line_count = 1;
            string line1, line25, line72;
            while (getline(actual, s)) {
                if (line_count == 1) line1 = s;
                if (line_count == 25) line25 = s;
                if (line_count == 72) line72 = s;
                line_count++;
            }
        
            REQUIRE(line1 == "6");
            REQUIRE(line25 == "0.8 0.8 0.8 0.8 0.8 0.4 ");
            REQUIRE(line72 == "0.5 0.5 0.5 0.5 0.5 0.5 ");
        }
    
        SECTION("Test that << operator correctly writes probability data for bigger images to file") {
            //write to file
            ifstream in(bigger_images);
            in >> training_data;
            TrainingModel training_model(training_data);
            ofstream out(output_file2);
            out << training_model;
            out.close();
            
            //read from file that was written to; check beginning, middle, and end
            ifstream actual(output_file2); string s;
            int line_count = 1;
            string line2, line33, line112;
            while (getline(actual, s)) {
                if (line_count == 2) line2 = s;
                if (line_count == 33) line33 = s;
                if (line_count == 112) line112 = s;
                line_count++;
            }
            
            REQUIRE(line2 == "10");
            REQUIRE(line33 == "0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 ");
            REQUIRE(line112 == "0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 ");
        }
    }
    
    TEST_CASE("Test that probabilities are read and stored in model correctly") {
        TrainingData training_data(training_labels);
        
        SECTION("Test that >> operator correctly loads class probabilities from file into model") {
            //create expected class probabilities
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
            
            //write to file
            ifstream in(training_images);
            in >> training_data;
            TrainingModel training_model(training_data);
            ofstream out(output_file);
            out << training_model;
            out.close();
            
            //read from file into an empty training model
            ifstream in2(output_file);
            TrainingModel training_model2;
            in2 >> training_model2;
            
            //get actual class probabilities
            vector<double> actual_class_probs = training_model2.GetClassProbs();
            
            REQUIRE(expected_class_probs == actual_class_probs);
        }
        
        SECTION("Test that >> operator correctly loads shaded probabilities from file into model") {
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
            
            // write to file
            ifstream in(training_images);
            in >> training_data;
            TrainingModel training_model(training_data);
            ofstream out(output_file);
            out << training_model;
            out.close();
            
            //read from file into an empty training model
            ifstream in2(output_file);
            TrainingModel training_model2;
            in2 >> training_model2;
            
            expected_probs.push_back(v0);
            expected_probs.push_back(v1);
            expected_probs.push_back(v2);
            expected_probs.push_back(v3);
            expected_probs.push_back(v9);
            
            //get actual output
            vector<vector<vector<double>>> actual_probs = training_model2.GetShadedProbs();
            
            REQUIRE(actual_probs[0][0][0] == Approx(expected_probs[0][0][0]).epsilon(0.001));
            REQUIRE(actual_probs[1][1][1] == Approx(expected_probs[1][1][1]).epsilon(0.001));
            REQUIRE(actual_probs[2][2][2] == Approx(expected_probs[2][2][2]).epsilon(0.001));
            REQUIRE(actual_probs[3][3][3] == Approx(expected_probs[3][3][3]).epsilon(0.001));
            REQUIRE(actual_probs[9][4][5] == Approx(expected_probs[4][4][5]).epsilon(0.001));
        }
    }
}