#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "core/classifier.h"

using namespace ci;
using namespace ci::app;

namespace naivebayes {
    class NaiveBayesAppGui : public App {
        public:
            /* Initializes classifier and image */
            void setup() override;
    
            /* Draws simulation every frame */
            void draw() override;

            /* Draws digit based on where mouse is being clicked */
            void mouseDown(MouseEvent event) override;
            void mouseDrag(MouseEvent event) override;

            /* Classifies or clears drawn image based on key pressed */
            void keyDown(KeyEvent event) override;
            
        private:
            //Classifier used to classify drawn digits; initialized with TrainingModel in setup()
            Classifier classifier_;
            
            //Digit drawn by user on GUI
            Image digit_;

            //Classification of drawn digit
            int classification_ = -1;
        
            //Percent of gui width + gui height that will be board width/height
            const float kBoardWidthPct = .25f;
            
            //Classifier was trained with 28x28 images
            const size_t kBoardDimension = 28;
        
            /* Drawing helper methods */
            void DrawTitle(int gui_width, int gui_height) const;
            void DrawSubtitle(int gui_width, int gui_height) const;
            void DrawBoard(int gui_width, int gui_height);
            void DrawClassification(int gui_width, int gui_height) const;
            void DrawPixels(const MouseEvent& event);
    };
}