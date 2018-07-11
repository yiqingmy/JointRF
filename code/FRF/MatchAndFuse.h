#pragma once
#ifndef MatchAndFuse_H
#define MatchAndFuse_H
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>

class MF {

public:
	MF() {}
	MF(cv::Mat _src_img);
	~MF() {}

public:
	void GetSaliencyMap(const cv::Mat& gray_img,cv::Mat& saliency_map,std::vector<cv::KeyPoint>& keypoints, cv::Mat& keyimg, const int octave, const int layer);
	void MatchMap(const MF& img, const MF& m_img, cv::Mat& s_matched_img, cv::Mat& i_matched_img, cv::Mat& pair_match, cv::Mat& fine_pairmatch, double& ratio, int& val_num, const int octave, const int layer, bool flag);
	void GetActivityMap(const std::vector<MF>& s_imgs,cv::Mat& activity_map,const int s,const bool button,const int r=2, const double eps=2);
	double right_ratio;


public:
	cv::Mat src_img;
	cv::Mat saliency_map;
	cv::Mat s_matched_map;
	cv::Mat i_matched_map;
	cv::Mat activity_map;
	cv::Mat key_map;
	std::vector<cv::KeyPoint>keypoints;
	static cv::Mat base_descriptor;
	static std::vector<cv::Mat> i_masks;
	static int tth;
	static int kkth;
	static double ave_time;
};
	


#endif