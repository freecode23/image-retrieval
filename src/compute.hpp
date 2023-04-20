//**********************************************************************************************************************
// FILE: compute.hpp
//
// DESCRIPTION
// Contains functions for pipeline
//
// AUTHOR
// Sherly Hartono
//**********************************************************************************************************************
#ifndef COMPUTE_H
#define COMPUTE_H
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "filter.hpp"
using namespace std;

enum feature_function{
  pixel_func,
  rgb_func,
  top_bom_func,
  rgb_mag_func,
  rgb_magori_func,
  rg_magori_func,
  rg_func
};



void show_img(cv::Mat img);


/*
  Given two feature vectors ft and fi, compute the ssd
  @params ft the target image that we want to match
  @params fi the image in database
 */

float compute_ssd(vector<float> &ft, vector<float> &fi);


/*
  Given two feature vectors ft and fi, compute the histogram intersection
  that is the minimum of the number of counts
  @params ft the target image that we want to match
  @params fi the image in database
 */
float compute_hist_intersect_error(vector<float> &ft, vector<float> &fi);


/*
  Given two feature vectors of multiple histo ft and fi, compute the histogram intersection
  by first splitting the ft to a and b parts then fi to a and b parts.
  512 is for 3D size histo of bin 8
  64 is for 2D size histo of bin 8
  a & b can be top (512) & bottom(512) or 
  a & b can be rgb (512) & magnitude(512) or
  a & b can be rgb (512) & magnitude-orientation (64) 
  @params size_a vector size a so we can split it
  @params ft the target image that we want to match
  @params fi the image in database
 */
float compute_mult_hist_intersect_error(vector<float> &ft, vector<float> &fi, int size_a, const float weight_a, const float weight_b);

/*
  Given a a dirPath argument, compute feature vector for all the images in that directory depending on the task number
  @params numOfArgs the number of arguments in argv
  @params dir_path_args the argument passed which is the directory path
  @params saveToFile the csv file name to save to
 */
void compute_fis(int num_of_args, char const *dir_path_args[], char *fi_csv, feature_function func);


/*
  RGB pixel
  Given an image, get 9 X 9 pixels of the center of the image of all the 3 channels
  @params img the image we want to compute feature vector of
  @params fx the resulting feature vector
 */
void compute_1_pixel(cv::Mat img, vector<float> &fx, int row_start, int col_start, int row_size, int col_size);

/*
  3D RGB Histo
  Given an image, get the normalzied histogram fx of the 3 color channel
  @params img the image we want to compute feature vector of
  @params fx the resulting feature vector
 */
void compute_2_rgb(cv::Mat img, vector<float> &fx);


/*
  top bottom 3D RGB histo
  Given an image, split the image to top and bottom half
  get two normalzied histogram of the 3 color channel
  @params img the image we want to compute feature vector of
  @params fx_top_bom_func the resulting feature vector
 */
void compute_3_top_bom_func(cv::Mat img, vector<float> &fx_top_bom_func);

/*
  3D RGB histo and 3D magnitude histo/vector
  Given an image get two feature vector appended together.
  whole image histo and gradient magnitude histo
  @params img the image we want to compute feature vector of
  @params fx_t_mag the resulting feature vector
 */
void compute_4_rgb_mag(cv::Mat img, vector<float> &fx_rgb_mag);


void compute_magnitude_orientation_hist(cv::Mat img, vector<float> &fx_grad_ori);

void compute_rg(cv::Mat img, vector<float> &fx);

void compute_5_rg_magori(cv::Mat img_uncropped, vector<float> &fx_rg_magori);

void compute_5_rgb_magori(cv::Mat img_uncropped, vector<float> &fx_rgb_magori);
/*
  Given a list of images and its fis and target image t, compute the top n most similar - minimum distance
  from ft
  @params name list of names of images
  @params fis vector of features of the images
 */
void compute_minimum_errors(vector<float> &ft, vector<vector<float>> &fis, vector<char *> &names, feature_function func);


/* Given :
  @params target image
  @params func the function to create vector
  @params filepath name of database fis
  It will: 
  - Computes the features for the target image by calling compute_featurex() 
  - reads the feature vector file by calling read_image_data_csv()
  and finally identifies the top N matches by calling compute_ranking()
*/
void get_top_n(cv::Mat t, char * fi_filepath, feature_function func);

#endif