#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "compute.hpp"
#include "filter.hpp"
#include "csv_util.h"
using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{

    // Define target image
    cv::Mat t;
    t = cv::imread("../olympus/pic.1016.jpg", 1);

    //  -----------------------------------------------------------------
    // Task 1
    // Part 1. Compute feature vector for database and save to a file
    cout << "\nCompute feature 1.." << endl;
    char fi1_csv[] = "../res/fi1.csv";
    compute_fis(argc, argv, fi1_csv, pixel_func);

    // Part 2. Given target image, the feature function enum, database fis
    // - computes the features for the target image, - reads the fis
    // - identifies the top N matches
    get_top_n(t, fi1_csv, pixel_func);


    //-----------------------------------------------------------------
    // Task 2
    cout << "\nCompute feature 2.." << endl;

    // Part 1. Compute feature vector for database and save to a file
    char fi2_csv[] = "../res/fi2.csv";
    compute_fis(argc, argv, fi2_csv, rgb_func);

    // Part 2.
    t = cv::imread("../olympus/pic.0164.jpg", 1);
    get_top_n(t, fi2_csv, rgb_func);

    //-----------------------------------------------------------------
    // Task 3
    char fi3_csv[] = "../res/fi3.csv";
    cout << "\nCompute feature 3.." << endl;
    compute_fis(argc, argv, fi3_csv, top_bom_func);

    t = cv::imread("../olympus/pic.0923.jpg", 1);
    get_top_n(t, fi3_csv, top_bom_func);

    //-----------------------------------------------------------------
    // Task 4
    cout << "\nCompute feature 4.." << endl;

    // 1. get fis
    char fi4_csv[] = "../res/fi4.csv";
    compute_fis(argc, argv, fi4_csv, rgb_mag_func);

    // 2. get top n
    t = cv::imread("../olympus/pic.1012.jpg", 1);
    get_top_n(t, fi4_csv, rgb_mag_func);


    //-----------------------------------------------------------------
    // Task 5
    cout << "\nCompute feature 5.." << endl;

    // 1. get fis
    char fi5_csv[] = "../res/fi5.csv";
    compute_fis(argc, argv, fi5_csv, rg_magori_func);

    //2. get top n
    t = cv::imread("../subset3/pic.0344.jpg", 1);
    get_top_n(t, fi5_csv, rg_magori_func); 
    // show_img(ti);
}
