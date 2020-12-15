#include "core/training_data.h"
#include "core/run_program.h"

int main(int argc, char* argv[]) {
    bool train_model = false; //train <images_file> <labels_file> <output_file> (4)
    bool load_model = false; //load <model_file> (2)
    bool load_and_test = false; //load_and_test <model_file> <test_images_file> <test_labels_file> (4)
    bool train_and_test = false; //train_and_test <training_images_file> <training_images_labels> <output_file> <test_images_file> <test_labels_file> (6)
    
    string training_images, training_labels, output_file, model_file, test_images, test_labels;
    vector<string> args(argv + 1, argv + argc);

    string options = "Please enter one of the following options: \n"
                     "1. train <images_file> <labels_file> <output_file> \n"
                     "2. load <model_file> \n"
                     "3. load_and_test <model_file> <test_images_file> <test_labels_file> \n"
                     "4. train_and_test <training_images_file> <training_images_labels> <output_file> <test_images_file> <test_labels_file>";

    auto print_options_invalid_num_args = [](string options) {
        cout << "Incorrect number of command line arguments entered. \n";
        cout << options << endl;
    };

    auto print_options_invalid_arg = [](string options) {
        cout << "Invalid command entered. \n";
        cout << options << endl;
    };

    if (args.size() != 2 && args.size() != 4 && args.size() != 6) {
        print_options_invalid_num_args(options);
    } else if (args.size() == 4) {
        if (args[0] == "train") {
            train_model = true;
            training_images = args[1];
            training_labels = args[2];
            output_file = args[3];
        } else if (args[0] == "load_and_test") {
            load_and_test = true;
            model_file = args[1];
            test_images = args[2];
            test_labels = args[3];
        } else {
            print_options_invalid_arg(options);
        }
    } else if (args.size() == 2) {
        if (args[0] == "load") {
            load_model = true;
            model_file = args[1];
        } else {
            print_options_invalid_arg(options);
        }
    } else if (args.size() == 6) {
        if (args[0] == "train_and_test") {
            train_and_test = true;
            training_images = args[1];
            training_labels = args[2];
            output_file = args[3];
            test_images = args[4];
            test_labels = args[5];
        } else {
            print_options_invalid_arg(options);
        }
    }

    if (train_model) {
        naivebayes::TrainModel(training_images, training_labels, output_file);
    } else if (load_model) {
        naivebayes::LoadModel(model_file);
    } else if (load_and_test) {
        naivebayes::LoadAndTest(model_file, test_images, test_labels);
    } else if (train_and_test) {
        naivebayes::TrainAndTest(training_images, training_labels, output_file, test_images, test_labels);
    }
}