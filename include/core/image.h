#pragma once

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace naivebayes {
    struct Image {
        /* Default constructor: used when reading in images from training file */
        Image() = default;
            
        /* Parameterized constructor: initializes image with the given 2d vector of pixels */
        explicit Image (vector<vector<int>>& image_pixels);

        vector<vector<int>> image_pixels_; //2d vector of pixels: 1 = shaded, 0 = unshaded
    
        /* Adds a row of chars (line) from a file into image_pixels_ */
        void AddPixelsRow(const string& pixels_row);
    };
}