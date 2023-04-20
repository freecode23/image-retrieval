//**********************************************************************************************************************
// FILE: compute.cpp
//
// DESCRIPTION
// Contains functions for different matching functions and pipelines
// AUTHOR
// Sherly Hartono
//**********************************************************************************************************************

#include "compute.hpp"
#include "csv_util.h"

float compute_ssd(vector<float> &ft, vector<float> &fi)
{

    // [ (x_1 - x_2) / stdev_x ] ^2
    float error = 0;
    for (int i = 0; i < ft.size(); i++)
    {
        error += (ft[i] - fi[i]) * (ft[i] - fi[i]);
    }
    return error;
}

float compute_hist_intersect_error(vector<float> &ft, vector<float> &fi)
{
    int i = 0;
    float similarity = 0;
    for (float count_per_bin_t : ft)
    {
        float count_per_bin_i = fi[i];
        similarity += min(count_per_bin_t, count_per_bin_i);
        i++;
        // cout << "similarity: " << similarity << endl;
    }
    return (1 - similarity);
}

float compute_mult_hist_intersect_error(vector<float> &ft, vector<float> &fi, int size_a, const float weight_a, const float weight_b)
{
    // split the ft and fi to a and b
    // a & b can be "top & bottom" or "rgb & magnitude" or "rgb & magnitudeorientation"
    // ft_a & ft_b
    vector<float> ft_a;
    // cout << "i: " << ft.size()<< endl;
    
    for (int i = 0; i < size_a; i++) // from 0 to end of ft_a
    {
        ft_a.push_back(ft[i]);
    }

    vector<float> ft_b;
    for (int i = size_a; i < ft.size(); i++) // from ft_a to the end
    {
        ft_b.push_back(ft[i]);
    }

    // fi a & b
    vector<float> fi_a;
    for (int i = 0; i < size_a; i++)
    {
        fi_a.push_back(fi[i]);
    }

    vector<float> fi_b;
    for (int i = size_a; i < fi.size(); i++)
    {
        fi_b.push_back(fi[i]);
    }

    float dist_a = compute_hist_intersect_error(ft_a, fi_a);
    float dist_b = compute_hist_intersect_error(ft_b, fi_b);
    float dist_ave = (weight_a * dist_a) + (weight_b * dist_b);
    return dist_ave;
}

void compute_1_pixel(cv::Mat img, vector<float> &fx, int row_start, int col_start, int row_size, int col_size)
{
    // 1. grab the feature vector of the 3D image
    for (int i = row_start; i < row_start + row_size; i++) // row
    {
        for (int j = col_start; j < col_start + col_size; j++) // column
        {
            for (int ch = 0; ch < 3; ch++) // channel
            {
                string col;
                if (ch == 0)
                {
                    col = "red";
                }
                else if (ch == 1)
                {
                    col = "green";
                }
                else if (ch == 2)
                {
                    col = "blue";
                }
                // cout << "i: " << i - centerRowStartIdx << " j: " << j - centerColStartIdx<< endl;
                // int val = img.at<cv::Vec3b>(i, j)[ch];
                // cout << col + ": " << val << endl;
                // insert the value to vector fx
                fx.push_back(img.at<cv::Vec3b>(i, j)[ch]);
            }
        }
    }
}

void compute_2_rgb(cv::Mat img, vector<float> &fx)
{
    // 1. create 3D histogram and initalize value to 0
    int Bsize = 8;
    for (int i = 0; i < Bsize * Bsize * Bsize; i++)
    {
        fx.push_back(0);
    }
    // 2. get a divisor
    const int divisor = 256 / Bsize;

    // 3. loop thru the image first 3 pixel and count their values
    for (int i = 0; i < img.rows; i++)
    { // row
        for (int j = 0; j < img.cols; j++)
        {                                       // column
            int R = img.at<cv::Vec3b>(i, j)[0]; // rgb values
            int G = img.at<cv::Vec3b>(i, j)[1];
            int B = img.at<cv::Vec3b>(i, j)[2];
            int r_idx = R / divisor; // rgb histo index
            int g_idx = G / divisor;
            int b_idx = B / divisor;

            int array_index = r_idx * Bsize * Bsize + g_idx * Bsize + b_idx;

            // increment the histo bin
            fx[array_index] += 1;

            // cout << "\ni: " << i << " j: " << j << endl;
            // cout << "R = " << R << " bin_idx = \t" << r_idx << endl;
            // cout << "G = " << G << " bin_idx = \t" << g_idx << endl;
            // cout << "B = " << B << " bin_idx = \t" << b_idx << endl;
            // cout << "array index: " << array_index << "\n";
        }
    }

    // 4. normalize histo
    float total_pixels = img.rows * img.cols; // 327,680 pixels
    int i = 0;
    for (float ele : fx)
    {
        if (ele > 0)
        {
            fx[i] = fx[i] / total_pixels;
            // printf("normalized count %.4f\n", fx[i]);
        }
        i += 1;
    }
}

void compute_3_top_bom(cv::Mat img, vector<float> &fx_top_bom)
{
    // 1. split image to two
    int y_bom = img.rows / 2;
    vector<float> fx_top;
    vector<float> fx_bom;
    cv::Mat img_top = img(cv::Rect(0, 0, img.cols, img.rows / 2));
    cv::Mat img_bom = img(cv::Rect(0, y_bom, img.cols, img.rows / 2));

    // 2. get two histo
    compute_2_rgb(img_top, fx_top); // 512
    compute_2_rgb(img_bom, fx_bom); // 512

    // 3. combine into 1 histo
    // append the bottom to top
    for (int i = 0; i < fx_bom.size(); i++)
    {
        fx_top.push_back(fx_bom[i]);
    }

    // 4. assign to our result
    fx_top_bom = fx_top;
}

void compute_4_rgb_mag(cv::Mat img, vector<float> &fx_rgb_mag)
{
    // fx_rgb + fx_mag = fx_rgb_mag
    // 1. get color histo: fx_rgb
    vector<float> fx_rgb;
    compute_2_rgb(img, fx_rgb); // fx_rgb.size() = 8 * 8 * 3 = 512

    // 2. get whole image magnitude fx_mag
    cv::Mat img_xdst;
    cv::Mat img_ydst;
    cv::Mat img_mag;
    sobelX3x3(img, img_xdst);
    sobelY3x3(img, img_ydst);
    magnitude(img_xdst, img_ydst, img_mag);

    // get magnitude histo: fx_mag
    vector<float> fx_mag;
    compute_2_rgb(img_mag, fx_mag);

    // 3. append mag to rgb
    for (int i = 0; i < fx_mag.size(); i++)
    {
        fx_rgb.push_back(fx_mag[i]);
    }
    // 4. get the final combined fx_rgb_mag
    fx_rgb_mag = fx_rgb; // size 1024
    // cout <<"fx size = " << fx_rgb_mag.size() << endl;
}

void compute_magnitude_orientation_hist(cv::Mat img, vector<float> &fx_magori)
{
    // 1. grey
    cv::Mat img_grey;
    greyscale(img, img_grey);

    // 2. sobel
    cv::Mat sx;
    cv::Mat sy;
    sobelX3x3(img_grey, sx);
    sobelX3x3(img_grey, sy);

    // 3. mag and orient
    cv::Mat mag_img;
    cv::Mat orient_img;
    magnitude(sx, sy, mag_img);
    orient(sx, sy, orient_img);

    // 4. get histo magori
    const int Bsize = 8;
    for (int i = 0; i < Bsize * Bsize; i++)
    {
        fx_magori.push_back(0);
    }

    const int divisor = 256 / Bsize;
    // 5. incement fx_magori histogram
    for (int i = 0; i < mag_img.rows; i++)
    {
        for (int j = 0; j < mag_img.cols; j++)
        {
            int M = mag_img.at<cv::Vec3b>(i, j)[0];    // magnitude value
            int O = orient_img.at<cv::Vec3b>(i, j)[0]; // orientation value

            int m_idx = M / divisor;
            int o_idx = O / divisor;
            int array_index = m_idx * Bsize + o_idx;
            fx_magori[array_index] += 1;
        }
    }

    // 6. normalize histo
    int total_pixels = mag_img.rows * mag_img.cols;
    float sum = 0;
    int i = 0;
    for (float ele : fx_magori)
    {
        if (ele > 0)
        {
            fx_magori[i] = fx_magori[i] / total_pixels;
            // printf("normalized count %.4f\n", fx_magori[i]);
            sum += fx_magori[i];
        }
        i += 1;
    }
    // printf("total sum should be 1: %.4f", sum);
}

void compute_rg(cv::Mat img, vector<float> &fx){

    // 1. create 2D histogram and initalize value to 0
    int Bsize = 8;
    for (int i = 0; i < Bsize * Bsize; i++)
    {
        fx.push_back(0);
    }
    // 2. get a divisor
    const int divisor = 256 / Bsize;

    // 3. loop thru the image first 3 pixel and count their values
    for (int i = 0; i < img.rows; i++)
    { // row
        for (int j = 0; j < img.cols; j++)
        {                                         // column
            float R = img.at<cv::Vec3b>(i, j)[0]; // rgb values
            float G = img.at<cv::Vec3b>(i, j)[1];
            float B = img.at<cv::Vec3b>(i, j)[2];

            // calculation to get index
            float r_pc = ((float) R) / (R + G + B);
            float g_pc = ((float) G) / (R + G + B);

            // alternative calculation to deal with divison by 0
            int r_idx = (Bsize * R) / (R + G + B + 1);
            int g_idx = (Bsize * G) / (R + G + B + 1);
            int array_index = r_idx * Bsize + g_idx;

            // increment the histo bin
            fx[array_index] += 1;

            // cout << "\ni: " << i << " j: " << j << endl;
            // cout << "R = " << R << " r_pc = "<< r_pc<< " r_idx = \t" << r_idx << endl;
            // cout << "G = " << G << " g_pc = "<< g_pc<< " g_idx = \t" << g_idx << endl;
            // cout << "array index: " << array_index << "\n";
        }
    }

    // 4. normalize histo
    float total_pixels = img.rows * img.cols; // 327,680 pixels
    int i = 0;
    for (float ele : fx)
    {
        if (ele > 0)
        {
            fx[i] = fx[i] / total_pixels;
            // printf("normalized count %.4f\n", fx[i]);
        }
        i += 1;
    }
}

void compute_5_rgb_magori(cv::Mat img_uncropped, vector<float> &fx_rgb_magori)
{
    // 1. crop image
    cv::Mat img = img_uncropped(cv::Rect(200, 200, 200, 100));

    // 2. compute fx_rgb
    vector<float> fx_rgb;
    compute_2_rgb(img, fx_rgb);

    // 3. compute fx_magori
    vector<float> fx_magori;
    compute_magnitude_orientation_hist(img, fx_magori);

    // 4. combined as 1 histo side by side
    // append magori to rgb
    for (int i = 0; i < fx_magori.size(); i++)
    {
        fx_rgb.push_back(fx_magori[i]);
    }

    // 5. get the final combined fx_rgb_mag
    fx_rgb_magori = fx_rgb;
    // cout <<"fx size = " << fx_magori.size() << endl;
    // cout <<"fx size = " << fx_rgb_magori.size() << endl;
}

void compute_5_rg_magori(cv::Mat img_uncropped, vector<float> &fx_rg_magori)
{
    // 1. crop image
    cv::Mat img = img_uncropped(cv::Rect(200, 200, 200, 100));

    // 2. compute fx_rg
    compute_rg(img, fx_rg_magori);

    // 3. compute fx_magori
    vector<float> fx_magori;
    compute_magnitude_orientation_hist(img, fx_magori);

    // 4. combined as 1 histo side by side
    // append magori to rgb
    for (int i = 0; i < fx_magori.size(); i++)
    {
        fx_rg_magori.push_back(fx_magori[i]);
    }
}

void compute_fis(int numOfArgs, char const *dir_path_args[], char *save_to_filepath, feature_function func)
{
    char dirpath[256];
    char fullPath[256];
    DIR *dirp;
    struct dirent *dp;

    // 1. if args is not sufficient exit
    if (numOfArgs < 2)
    {
        printf("usage: %s <directory path>\n", dir_path_args[0]);
        exit(-1);
    }

    // 2. get the directory path
    strcpy(dirpath, dir_path_args[1]);
    printf("Processing directory %s\n", dirpath);

    // 3. open the directory
    dirp = opendir(dirpath);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirpath);
        exit(-1);
    }

    int idx = 0;
    // 4. loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL)
    {

        char *image_name = dp->d_name;
        // 5. check if the file is an image
        if (strstr(image_name, ".jpg") ||
            strstr(image_name, ".png") ||
            strstr(image_name, ".ppm") ||
            strstr(image_name, ".tif"))
        {

            // printf("processing image file: %s\n", image_name);
            // printf("full path name: %s\n", fullPath);
            // 6. build the overall filename
            strcpy(fullPath, dirpath);
            strcat(fullPath, "/");
            strcat(fullPath, image_name);

            // 7. get image in the directory
            cv::Mat i = cv::imread(fullPath, 1);
            // 8. compute the feature 1 for this image
            vector<float> fi;
            if (func == pixel_func)
            {
                // 1. get the index of center row's top left corner
                int row_start = (i.rows / 2) - 4;
                int col_start = (i.cols / 2) - 4;

                // 2. 9X9 pixel
                int pixel_size = 9;
                compute_1_pixel(i, fi, row_start, col_start, pixel_size, pixel_size);
            }
            else if (func == rgb_func)
            {
                compute_2_rgb(i, fi);
            }
            else if (func == top_bom_func)
            {
                compute_3_top_bom(i, fi);
            }
            else if (func == rgb_mag_func)
            {
                compute_4_rgb_mag(i, fi);
            }
            else if (func == rgb_magori_func)
            {
                compute_5_rgb_magori(i, fi);
            }
            else if (func == rg_magori_func)
            {
                compute_5_rg_magori(i, fi);
            }
            else if (func == rg_func)
            {
                compute_rg(i, fi);
            }

            int overwrite = 0;
            if (idx == 0)
            {
                overwrite = 1;
            }
            // 9. save the feature to a new csv path
            append_image_data_csv(save_to_filepath, image_name, fi, overwrite);
            idx += 1;
        }
    }
    cout << "finish compute fis" << endl;
}

void compute_minimum_errors(vector<float> &ft, vector<vector<float>> &fis, vector<char *> &names, feature_function func)
{
    // 1. Get list of errors for each feature
    vector<float> error_list;

    int rgb_histo_size = 512; // 8 bins^3 channel
    // 2. for each fis compute distance from ft
    for (int i = 0; i < fis.size(); i++)
    {
        vector<float> fi;
        // grab a single fi vector

        for (int j = 0; j < fis[i].size(); j++)
        {
            fi.push_back(fis[i][j]);
            // printf("%.4f  ", result_fis[i][j]);
        }
        float error;

        if (func == pixel_func)
        {
            error = compute_ssd(ft, fi);
        }
        else if (func == rgb_func)
        {
            error = compute_hist_intersect_error(ft, fi);
        }
        else if (func == top_bom_func)
        {
            const float top_weight = 0.2;
            const float bom_weight = 0.8;
            error = compute_mult_hist_intersect_error(ft, fi, rgb_histo_size, top_weight, bom_weight);
        }
        else if (func == rgb_mag_func)
        {
            const float rgb_weight = 0.7;
            const float texture_weight = 0.3;
            error = compute_mult_hist_intersect_error(ft, fi, rgb_histo_size, rgb_weight, texture_weight);
        }
        else if (func == rgb_magori_func)
        {
            const float rgb_weight = 0.8;
            const float texture_weight = 0.2;
            error = compute_mult_hist_intersect_error(ft, fi, rgb_histo_size, rgb_weight, texture_weight);
        }
        else if (func == rg_magori_func)
        {
            const float rg_weight = 0.8;
            const float texture_weight = 0.2;
            int rg_histo_size = 64; // 8 bins^2 channel
           error = compute_mult_hist_intersect_error(ft, fi, rg_histo_size, rg_weight, texture_weight);
        }
        else if (func == rg_func)
        {
           error = compute_hist_intersect_error(ft, fi);
        }
        error_list.push_back(error);
    }

    // 3. get top 3 minimum distance
    vector<char *> top3;
    int i = 0;
    while (i < 10)
    {
        // get minimum value in error list
        int min_ele_idx = std::min_element(error_list.begin(), error_list.end()) - error_list.begin();
        int min_ele_val = *std::min_element(error_list.begin(), error_list.end());
        cout << "\n"<< i + 1 << ": ";
        cout << names.at(min_ele_idx) << endl;
        cout << "error: " << error_list.at(min_ele_idx) << endl;

        // remove this index from error_list
        error_list.erase(error_list.begin() + min_ele_idx);

        // add name from result_name at this index to top3
        top3.push_back(names.at(min_ele_idx));

        // remove it from the result name
        names.erase(names.begin() + min_ele_idx);
        i++;
    }
}

void get_top_n(cv::Mat t, char *fi_filepath, feature_function func)
{
    // 1. get ft
    vector<float> ft;
    if (func == pixel_func)
    {
        // 1. get the index of center row's top left corner
        int row_start = (t.rows / 2) - 4;
        int col_start = (t.cols / 2) - 4;

        // 2. 9X9 pixel
        int pixel_size = 9;
        compute_1_pixel(t, ft, row_start, col_start, pixel_size, pixel_size);
    }
    else if (func == rgb_func)
    {
        compute_2_rgb(t, ft);
    }
    else if (func == top_bom_func)
    {
        compute_3_top_bom(t, ft); // ft top and bottom
    }
    else if (func == rgb_mag_func)
    {
        compute_4_rgb_mag(t, ft);
    }
    else if (func == rgb_magori_func)
    {
        compute_5_rgb_magori(t, ft);
    }
    else if (func == rg_magori_func)
    {
        compute_5_rg_magori(t, ft);
    }
    else if (func == rg_func)
    {
        compute_rg(t, ft);
    }

    // 2. get fis and their file names
    vector<char *> result_name;
    vector<vector<float>> result_fis;
    read_image_data_csv(fi_filepath, result_name, result_fis, 1);
    cout << "finsih read image" << endl;

    // 3. calculate rank
    compute_minimum_errors(ft, result_fis, result_name, func);
}

void show_img(cv::Mat img)
{
    cv::imshow("img", img);
    while (1)
    {
        cv::imshow("img", img);
        int k = cv::waitKey(0);
        if (k == 113)
        {
            break;
        }
    }
}
