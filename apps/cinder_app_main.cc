#include "gui/naive_bayes_app_gui.h"

using naivebayes::NaiveBayesAppGui;

void prepareSettings(NaiveBayesAppGui::Settings* settings) {
    settings->setWindowSize(1300, 900);
    settings->setFrameRate(60.0f);
}

// This line is a macro that expands into an "int main()" function.
CINDER_APP(NaiveBayesAppGui, ci::app::RendererGl, prepareSettings)