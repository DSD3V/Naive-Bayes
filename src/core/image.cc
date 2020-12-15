#include "core/image.h"

namespace naivebayes {
    Image::Image(vector<vector<int>>& image_pixels) : image_pixels_(image_pixels) {}

    void Image::AddPixelsRow(const string& pixels_row) {
        vector<int> pixels_row_v;
        for (const char pixel : pixels_row) {
            pixels_row_v.emplace_back((pixel == ' ') ? 0 : 1); //0 if unshaded, 1 if shaded
        }
        image_pixels_.push_back(pixels_row_v);
    }
}