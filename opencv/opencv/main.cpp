#include <opencv2/opencv.hpp>  
#include <iostream>  
using namespace std;
using namespace cv;
int main()
{
	//��ȡ���ص�һ��ͼƬ����ʾ����  
	Mat img = imread("D:\\2.jpg");
	imshow("MM Viewer", img);
	//�ȴ��û�����  
	waitKey(0);
	return 0;
}