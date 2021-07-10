#include "gui/naive_bayes_app_gui.h"

namespace naivebayes {
    void NaiveBayesAppGui::setup() {
        //Initializes classifier using already trained TrainingModel
        TrainingModel* training_model = new TrainingModel;
        ifstream in("data/training_model.txt");
        in >> *training_model;
        classifier_ = Classifier(*training_model);
        
        //Initializes image; starts off as all white pixels
        for (size_t i = 0; i < kBoardDimension; ++i) {
            vector<int> pixels_row;
            for (size_t j = 0; j < kBoardDimension; ++j) {
                pixels_row.push_back(0);
            }
            digit_.image_pixels_.push_back(pixels_row);
        }
    }
    
    void NaiveBayesAppGui::draw() {
        gl::clear();
        int gui_width = getWindowWidth(), gui_height = getWindowHeight();
        
        DrawTitle(gui_width, gui_height);
        DrawSubtitle(gui_width, gui_height);
        DrawBoard(gui_width, gui_height);
        DrawClassification(gui_width, gui_height);
    }
    
    void NaiveBayesAppGui::DrawTitle(const int gui_width, const int gui_height) const {
        const Font title_font = Font("Roboto", (gui_width + gui_height) * .023f);
        gl::drawStringCentered("Naïve Bayes Digit Classifier", glm::vec2(gui_width / 2, gui_height * .02f),
                               Color(1, 1, 1), title_font);
    }

    void NaiveBayesAppGui::DrawSubtitle(const int gui_width, const int gui_height) const {
        const Font subtitle_font = Font("Roboto", (gui_width + gui_height) * .017f);
        gl::drawStringCentered(R"(Draw a digit and press "c" to classify it, and "e" to erase it.)",
                               glm::vec2(gui_width / 2, gui_height * .1f),Color(1, 1, 1), subtitle_font);
    }

    void NaiveBayesAppGui::DrawBoard(const int gui_width, const int gui_height) {
        float board_width = kBoardWidthPct * (gui_width + gui_height);
        float cell_width = board_width / kBoardDimension;
        
        glm::vec2 board_top_left = glm::vec2((gui_width / 2) - (board_width / 2), (gui_height / 2) - (board_width / 2));
        glm::vec2 board_bottom_right = glm::vec2(board_top_left.x + board_width, board_top_left.y + board_width);
        
        float original_x = board_top_left.x;
        glm::vec2 cell_top_left = board_top_left;
        
        gl::drawSolidRect(Rectf(board_top_left, board_bottom_right));
        
        for (size_t i = 0; i < kBoardDimension; ++i) {
            for (size_t j = 0; j < kBoardDimension; ++j) {
                Rectf r(cell_top_left, glm::vec2(cell_top_left.x + cell_width, cell_top_left.y + cell_width));
                gl::color(0, 0, 0);
                if (digit_.image_pixels_[i][j] == 0) {
                    gl::drawStrokedRect(r, 1);
                } else if (digit_.image_pixels_[i][j] == 1) {
                    gl::drawSolidRect(r);
                }
                gl::color(1, 1, 1);
                cell_top_left.x += cell_width;
            }
            cell_top_left.x = original_x;
            cell_top_left.y += cell_width;
        }
    }

    void NaiveBayesAppGui::DrawClassification(const int gui_width, const int gui_height) const {
        const Font classification_font = Font("Roboto", (gui_width + gui_height) * .017f);
        
        const string classification = (classification_ == -1 ? "_" : std::to_string(classification_));
        
        gl::drawStringCentered("Classification: " + classification, glm::vec2(gui_width / 2, gui_height * .91f),
                               Color(1, 1, 1), classification_font);
    }

    void NaiveBayesAppGui::mouseDown(MouseEvent event) { DrawPixels(event); }
    void NaiveBayesAppGui::mouseDrag(MouseEvent event) { DrawPixels(event); }
    
    void NaiveBayesAppGui::DrawPixels(const MouseEvent& event) {
        int gui_width = getWindowWidth(), gui_height = getWindowHeight();
        float board_width = kBoardWidthPct * (gui_width + gui_height);
        float pixel_width = board_width / kBoardDimension;

        glm::vec2 board_top_left = glm::vec2((gui_width / 2) - (board_width / 2), (gui_height / 2) - (board_width / 2));
        glm::vec2 board_bottom_right = glm::vec2(board_top_left.x + board_width, board_top_left.y + board_width);

        int x = event.getPos().x;
        int y = event.getPos().y;

        if (x > board_top_left.x && x < board_bottom_right.x && y > board_top_left.y && y < board_bottom_right.y) {
            float board_x = x - board_top_left.x;
            float board_y = y - board_top_left.y;

            int image_x = (int) (board_x / pixel_width);
            int image_y = (int) (board_y / pixel_width);

            digit_.image_pixels_[image_y][image_x] = 1;
        }
    }

    void NaiveBayesAppGui::keyDown(KeyEvent event) {
        switch (event.getCode()) {
            case KeyEvent::KEY_c:
                //classify the image that is currently drawn on the GUI
                classification_ = classifier_.Classify(digit_);
                break;
            case KeyEvent::KEY_e:
                //clear the drawn image and classification
                for (size_t i = 0; i < kBoardDimension; ++i) {
                    for (size_t j = 0; j < kBoardDimension; ++j) {
                        digit_.image_pixels_[i][j] = 0;
                    }
                }
                classification_ = -1;
                break;
        }
    }
}