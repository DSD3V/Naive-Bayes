#include "core/training_data.h"
#include <catch2/catch.hpp>

namespace naivebayes {
    const string training_images = "tests_data/test_trainingimages.txt";
    const string bigger_images = "tests_data/test_trainingimages_bigger.txt";
    const string training_labels = "tests_data/test_traininglabels.txt";

    TEST_CASE("Test that image file is parsed correctly") {
        SECTION("Test that >> operator correctly stores image data") {
            ifstream in(training_images);
            TrainingData training_data(training_labels);
            in >> training_data;
        
            //create expected output of vector of images
            vector<vector<Image>> expected_images;
        
            vector<Image> images1;
        
            Image image1_1;
            string row1 = " ###  ";
            string row2 = " ###  ";
            string row3 = " ###  ";
            string row4 = " ###  ";
            string row5 = " ###  ";
            string row6 = "      ";
            image1_1.AddPixelsRow(row1);
            image1_1.AddPixelsRow(row2);
            image1_1.AddPixelsRow(row3);
            image1_1.AddPixelsRow(row4);
            image1_1.AddPixelsRow(row5);
            image1_1.AddPixelsRow(row6);
        
            Image image1_4;
            row1 = "  #   ";
            row2 = " ##   ";
            row3 = " ##   ";
            row4 = "  #   ";
            row5 = "##### ";
            row6 = "      ";
            image1_4.AddPixelsRow(row1);
            image1_4.AddPixelsRow(row2);
            image1_4.AddPixelsRow(row3);
            image1_4.AddPixelsRow(row4);
            image1_4.AddPixelsRow(row5);
            image1_4.AddPixelsRow(row6);
        
            images1.push_back(image1_1);
            images1.push_back(image1_4);
        
            vector<Image> images2;
        
            Image image2_2;
            row1 = "##### ";
            row2 = "   ## ";
            row3 = " ###  ";
            row4 = "#     ";
            row5 = " #### ";
            row6 = "      ";
        
            image2_2.AddPixelsRow(row1);
            image2_2.AddPixelsRow(row2);
            image2_2.AddPixelsRow(row3);
            image2_2.AddPixelsRow(row4);
            image2_2.AddPixelsRow(row5);
            image2_2.AddPixelsRow(row6);
        
            images2.push_back(image2_2);
        
            vector<Image> images3;
        
            Image image3_3;
            row1 = "##### ";
            row2 = "  ### ";
            row3 = "##### ";
            row4 = "  ### ";
            row5 = "####  ";
            row6 = "      ";
        
            image3_3.AddPixelsRow(row1);
            image3_3.AddPixelsRow(row2);
            image3_3.AddPixelsRow(row3);
            image3_3.AddPixelsRow(row4);
            image3_3.AddPixelsRow(row5);
            image3_3.AddPixelsRow(row6);
        
            images3.push_back(image3_3);
        
            expected_images.push_back(images1);
            expected_images.push_back(images2);
            expected_images.push_back(images3);
        
        
            //get actual output
            vector<vector<Image>> actual_images = training_data.GetTrainingDataImages();
        
            REQUIRE(actual_images[1][0].image_pixels_ == expected_images[0][0].image_pixels_);
            REQUIRE(actual_images[1][3].image_pixels_ == expected_images[0][1].image_pixels_);
            REQUIRE(actual_images[2][1].image_pixels_ == expected_images[1][0].image_pixels_);
            REQUIRE(actual_images[3][2].image_pixels_ == expected_images[2][0].image_pixels_);
        }
    
        SECTION("Test that >> operator correctly stores image data for bigger images") {
            ifstream in(bigger_images);
            TrainingData training_data(training_labels);
            in >> training_data;
            
            //create expected output of vector of images
            vector<vector<Image>> expected_images;
            
            vector<Image> images3;
            
            Image image3;
            string row1 = "          ";
            string row2 = "  #####   ";
            string row3 = "     ###  ";
            string row4 = "     ###  ";
            string row5 = "  #####   ";
            string row6 = "     ###  ";
            string row7 = "     ###  ";
            string row8 = "     ##   ";
            string row9 = "  #####   ";
            string row10 = "          ";
            
            image3.AddPixelsRow(row1);
            image3.AddPixelsRow(row2);
            image3.AddPixelsRow(row3);
            image3.AddPixelsRow(row4);
            image3.AddPixelsRow(row5);
            image3.AddPixelsRow(row6);
            image3.AddPixelsRow(row7);
            image3.AddPixelsRow(row8);
            image3.AddPixelsRow(row9);
            image3.AddPixelsRow(row10);
            
            images3.push_back(image3);
            
            expected_images.push_back(images3);
            
            //get actual output
            vector<vector<Image>> actual_images = training_data.GetTrainingDataImages();
            
            REQUIRE(actual_images[3][1].image_pixels_ == expected_images[0][0].image_pixels_);
        }
    }
}