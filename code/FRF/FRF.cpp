// FRF.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
#include"stdafx.h"
#include"MatchAndFuse.h"
#include<fstream>
#include<Windows.h>

#define _DEBUG
#define BASE_SIZE 100
#define OCTAVE_NUM 8
#define LAYER_NUM 8
using namespace std;
using namespace cv;

bool FindOrginalImages(const std::string& folder, std::vector<std::string>& validImageFiles, const std::string& format, std::vector<std::string>& filename);
Mat MF::base_descriptor = cv::Mat::zeros(cv::Size(), CV_32FC1);
vector<Mat> MF::i_masks = vector<Mat>();


int main(int arg, char* argv[])
{


	double T1 = (double)cv::getTickCount();
	string folder = argv[1];
	string format = argv[2];
	vector<string> ImageFiles;
	static vector<std::string> filename;
	if (!FindOrginalImages(folder, ImageFiles, format, filename)) {
		std::cout << "No file detected, images reading error!" << std::endl;
	}

	//Get the saliency map and keypoints of each image.

	int max = 0;
	MF base_img;
	vector<MF> MF_imgs;

	//-----------------------
	int oth = 4;
	int lth = 2;
	//-----------------------

	int fp = 0;//Record the position of the base image.
	for (size_t i = 0; i < ImageFiles.size(); i++) {
		Mat src = imread(ImageFiles[i]);
		MF *mf_img = new MF(src);
		(*mf_img).GetSaliencyMap((*mf_img).src_img, (*mf_img).saliency_map, (*mf_img).keypoints, (*mf_img).key_map, oth, lth);
		if ((*mf_img).keypoints.size()>max) {//Reserve the basic information by finding the image with the most keypoints.
			max = (*mf_img).keypoints.size();
			base_img = (*mf_img);
			fp = i;
		}
		MF_imgs.push_back(*mf_img);
		delete(mf_img);
	}

	MF_imgs.erase(MF_imgs.begin() + fp);
	MF_imgs.insert(MF_imgs.begin(), base_img);

	//Get the matched maps from the saliency maps.
	//----------------------------------------
	double ratio_sum = 0;
	//----------------------------------------
	for (size_t j = 0; j < MF_imgs.size(); j++) {
		MF& s_img = MF_imgs[j];
		vector<cv::KeyPoint>& keypoints = MF_imgs[j].keypoints;

		if (keypoints.size() == 0) {
			MF_imgs[j].s_matched_map = s_img.saliency_map;
			MF_imgs[j].i_matched_map = s_img.src_img;
		}
		//--------------------
		Mat pair_match;
		Mat fine_pair_match;
		double ratio;
		int val_num;
		//--------------------
		MF_imgs[j].MatchMap(s_img, base_img, MF_imgs[j].s_matched_map, MF_imgs[j].i_matched_map, pair_match, fine_pair_match, ratio, val_num, oth, lth, j);

	}

	//Get the activity map.
	for (size_t adx = 0; adx < MF_imgs.size(); adx++) {
		MF_imgs[adx].GetActivityMap(MF_imgs, MF_imgs[adx].activity_map, adx, adx);
	}

	//Fuse the images.
	vector<Mat>f_imgs;
	Mat combine_img = Mat::zeros(Size(MF_imgs[0].src_img.cols, MF_imgs[0].src_img.rows), CV_32FC3);
	Mat sum_weight = Mat::zeros(Size(MF_imgs[0].src_img.cols, MF_imgs[0].src_img.rows), CV_32FC3);
	for (size_t fdx = 0; fdx < MF_imgs.size(); fdx++) {
		Mat src = MF_imgs[fdx].i_matched_map;
		src.convertTo(src, CV_32F);
		Mat mask = MF_imgs[fdx].activity_map;
		Mat temp_mask;
		if (src.channels() == 3) {
			cvtColor(mask, temp_mask, CV_GRAY2RGB);
		}
		else {
			temp_mask = mask;
		}
		temp_mask.convertTo(temp_mask, CV_32F);
		combine_img = combine_img + src.mul(temp_mask);
		sum_weight = sum_weight + temp_mask;
	}
	Mat fused_img = combine_img.mul(1. / sum_weight);
	double T2 = (double)cv::getTickCount();
	double T_time = ((double(T2 - T1)) / cv::getTickFrequency());
	//f_ratio << "Running time:" << T_time << endl;
	cout << "Running time:" << T_time << endl;


	MF_imgs.clear();

	//}
	//}
	// f_ratio.close();

	system("pause");
	return 0;
}

//Read all the images under the same folder
bool FindOrginalImages(const std::string& folder, std::vector<std::string>& validImageFiles, const std::string& format, std::vector<std::string>& filename) {
	WIN32_FIND_DATAA findFileData;
	std::string path = folder + "/*." + format;
	HANDLE hFind = FindFirstFileA((LPCSTR)path.c_str(), &findFileData);
	if (hFind == INVALID_HANDLE_VALUE)
		return false;

	do {
		std::string fileName = findFileData.cFileName;
		filename.push_back(fileName);
		std::string ext = fileName.substr(fileName.find_last_of('.') + 1, fileName.length());

		if (ext == format) {
			std::string tmpFileName = folder + '\\' + fileName;
			validImageFiles.push_back(tmpFileName);
		}

	} while (FindNextFileA(hFind, &findFileData));

	return true;
}