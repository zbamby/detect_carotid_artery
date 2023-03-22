#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

// 全局变量定义
const std::string VIDEO_SRC = "./detect_carotid_artery/230213_1.mp4"; 
const std::string VIDEO_OUT = "./detect_carotid_artery/230213_1_out.avi";
const std::string WINDOW_1 = "origin_video"; // 原视频
const std::string WINDOW_2 = "process_filter"; // 滤波后视频
const std::string WINDOW_3 = "process_detect"; // 检测后视频
const std::string WINDOW_4 = "process_extract"; // 轮廓提取后 ROI_rec区域 视频

cv::VideoCapture g_video_cap; 
cv::VideoWriter g_video_writer;


int g_slider_position = 65; // 视频开始播放的位置设置
int g_slider_run = 1; //<0: run mode; 1: single mode; 0: stop mode
int g_slider_dontset = 0;

cv::Mat g_src_img; // 源图
cv::Mat	g_ROI_img; // 600X800 灰度图
cv::Mat g_filtered_img; // g_ROI_img 滤波后图像
cv::Mat	g_ROI_rec_img; // 600X800ROI 中的 ROI_rec矩形探测区域
cv::Mat g_filtered_ROI_rec_binary_img; // ROI_rec矩形区图像 二值化结果

int g_nFilteropt = 3; // 滤波类型
int g_nFiltervalue = 1; // 滤波参数，当为方框、均值、高斯滤波时，表示核的大小

int g_ROI_rec_nThresholdValue = 100; // 二值化的阈值
int g_ROI_rec_nThresholdType = 0;	// 二值化阈值类型 0：CV_THRESH_BINARY

int g_ROI_rec_naThresholdMethod = CV_ADAPTIVE_THRESH_MEAN_C; // 自适应阈值算法
int g_ROI_rec_naThresholdBlocksize = 3; // 用于计算阈值的邻域尺寸
int g_ROI_rec_naThresholdC = 0; // 自适应阈值的常数，不知道干嘛的




const int ROI_WIDTH = 600;
const int ROI_HIGHT = 800;

class HoughCircleVarible
{
public:
	int HC_dp = 1;
	int HC_minDist = 500;
	int HC_canny = 30;
	int HC_CTh = 25;
	int HC_minR = 40;
	int HC_maxR = 72;
};
HoughCircleVarible g_HoughVar;
const int CIR_DIS_TH = 20; // 当上一检出圆和当前检出圆圆心距离超过此极限值时，认为检出失败

// 鼠标回调函数全局变量声明
cv::Rect g_ROI_rec = cv::Rect(0, 0, ROI_WIDTH, ROI_HIGHT); // 600X800 中的矩形区域，初始化为整个区域
bool g_bDrawingBox = false; // 是否进行绘制

int n_detect_start = 0; // 检出的第一帧的帧位
int n_detect_suc = 0; // 从 n_detect_start 开始的检出帧数
int n_detect_fail = 0; // 从 n_detect_start 开始的未检出帧数
int n_detect_fail_interval = 0; // 连续未检出帧数
int n_detect_fail_interval_Max = 0; //连续未检出帧数的最大值
std::vector<cv::Vec3f> q_Circles; // 存放合格检出圆
std::vector<std::vector<std::vector<cv::Point>>> q_Contours; // 存放合格检出圆确定的 ROI_rec 中检测出的轮廓
const int REC_CIR_D = 20; // 人为设定的 ROI_rec 对于 合格检出圆 的边界延伸区参数
const int N_COUNT_FAIL_INTERVAL = 100; // 人为设定的 最大连续未检出帧数 阈值，超过此数则认为目标跟随失败，程序结束




void onTrackBarSlide(int pos, void*);
cv::Mat img_filter(int opt, const cv::Mat& image);
static void on_Filter(int ,void*);
void ProSimple(const cv::Mat& image);
void imgDFT(const cv::InputArray src_ROI);
void imgHoughLines(const cv::InputOutputArray srcImg);
void imgHoughLinesP(const cv::InputOutputArray srcImg);
static void on_HoughCircles(int, void*);
static void on_HoughCircles1(int, void*);

void imgHoughCircles();
static void on_MouseHandle(int event, int x, int y, int flags, void*);
static void on_MouseHandle1(int event, int x, int y, int flags, void*);
double dis_between_points(cv::Point& A, cv::Point& B);

void on_ExtractContours(int, void*); // 简单阈值分割回调函数


int main()
{

	//读入视频
	// cv::namedWindow(WINDOW_1, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_2, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_3, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_4, CV_WINDOW_AUTOSIZE);
	g_video_cap.open(VIDEO_SRC);
	if (!g_video_cap.isOpened()) return -1;
	
	// 视频基本信息
	int frames = (int)g_video_cap.get(cv::CAP_PROP_FRAME_COUNT);
	int tmpw = (int)g_video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int tmph = (int)g_video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Video has " << frames << " frames of dimensions("
		<< tmpw << ", " << tmph << ")." << std::endl;
	g_video_cap.set(cv::CAP_PROP_POS_FRAMES, g_slider_position);

	//创建输出对象
	double fps = g_video_cap.get(cv::CAP_PROP_FPS);
	cv::Size size_ROI(((ROI_WIDTH+1)/2+1)/2, ((ROI_HIGHT + 1) / 2 + 1) / 2);
	g_video_writer.open(VIDEO_OUT, CV_FOURCC('M', 'J', 'P', 'G'), fps, size_ROI,false);

	//创建滑动条
	cv::createTrackbar("Position", WINDOW_2, &g_slider_position, frames,
		onTrackBarSlide); // 播放滑动条
	cv::createTrackbar("KernelSize", WINDOW_2, &g_nFiltervalue, 40, on_Filter); // 滤波滑动条
	cv::createTrackbar("FilterType", WINDOW_2, &g_nFilteropt, 5, on_Filter); // 滤波类型
	cv::createTrackbar("b_Th_Value", WINDOW_2, &g_ROI_rec_nThresholdValue, 255, on_ExtractContours); 
	cv::createTrackbar("b_Th_Type", WINDOW_2, &g_ROI_rec_nThresholdType, 4, on_ExtractContours);
	cv::createTrackbar("b_aTh_Meth", WINDOW_2, &g_ROI_rec_naThresholdMethod, 1, on_ExtractContours);
	cv::createTrackbar("b_aTh_blockSize", WINDOW_2, &g_ROI_rec_naThresholdBlocksize, 7, on_ExtractContours);
	cv::createTrackbar("b_aTh_C", WINDOW_2, &g_ROI_rec_naThresholdC, 255, on_ExtractContours);

	// 霍夫圆检测相关参数滑动条
	cv::createTrackbar("HC_dp", WINDOW_3, &(g_HoughVar.HC_dp), 5, on_HoughCircles); // 累加器图像的分辨率
	cv::createTrackbar("HC_minDist", WINDOW_3, &(g_HoughVar.HC_minDist), 1000, on_HoughCircles); //可检测的最小圆距离
	cv::createTrackbar("HC_canny", WINDOW_3, &(g_HoughVar.HC_canny), 200, on_HoughCircles); // canny边缘检测高阈值
	cv::createTrackbar("HC_cirTh", WINDOW_3, &(g_HoughVar.HC_CTh), 60, on_HoughCircles); // 圆心累加器阈值
	cv::createTrackbar("HC_minR", WINDOW_3, &(g_HoughVar.HC_minR), 100, on_HoughCircles); // 可检测的最小圆半径
	cv::createTrackbar("HC_maxR", WINDOW_3, &(g_HoughVar.HC_maxR), 100, on_HoughCircles); // 可检测的最大圆半径


	
	//定义感兴趣的区域范围 
	cv::Mat src_ROI;
	cv::Rect rect(250, 110, ROI_WIDTH, ROI_HIGHT);
	cv::setMouseCallback(WINDOW_3, on_MouseHandle1); // 注册鼠标回调函数

	while (1)
	{
		if (g_slider_run != 0) {
			// 读取一帧图像
			g_video_cap >> g_src_img; if (!g_src_img.data) break;
			src_ROI = g_src_img(rect);
			cvtColor(src_ROI, g_ROI_img, cv::COLOR_BGR2GRAY); // 转成灰度图

			//更新滑动条的位置
			int current_pos = (int)g_video_cap.get(cv::CAP_PROP_POS_FRAMES);
			g_slider_dontset = 1;
			cv::setTrackbarPos("Position", WINDOW_2, current_pos);

			////显示原始感兴趣区域
			//cv::imshow(WINDOW_1, src_ROI);
			

			// 霍夫圆检测
			//imgHoughCircles();
			if (n_detect_fail_interval > N_COUNT_FAIL_INTERVAL)
			{
				std::cout << "第" << int(g_video_cap.get(cv::CAP_PROP_POS_FRAMES)) << "时目标跟随失败，" << std::endl <<
					"共检测" << int(g_video_cap.get(cv::CAP_PROP_POS_FRAMES)) - n_detect_start << "帧" << std::endl <<
					"合格检出圆为\t" << n_detect_suc << std::endl <<
					"未检出帧数为\t" << n_detect_fail << std::endl <<
					"检出率为\t" << n_detect_suc / (double(g_video_cap.get(cv::CAP_PROP_POS_FRAMES)) - n_detect_start) << std::endl <<
					"最大检测失败连续帧数为\t" << n_detect_fail_interval_Max << std::endl;
				return -1;
			}
			on_HoughCircles1(0, 0);
			 

			g_slider_run -= 1;
		}

		// 一些键盘控制的播放设置
		char c = (char)cv::waitKey(10);
		if (c == 's') // single step
		{
			g_slider_run = 1;
			std::cout << "Single step, run = " << g_slider_run << std::endl;
		}
		if (c == 'r')//run mode
		{
			g_slider_run = -1;
			std::cout << "Run mode, run = " << g_slider_run << std::endl;
		}
		if (c == 27)
			break;
	}

	std::cout << 
		"共检测" << std::min(374,int(g_video_cap.get(cv::CAP_PROP_POS_FRAMES))) - n_detect_start << "帧" << std::endl <<
		"合格检出圆为\t" << n_detect_suc << std::endl <<
		"未检出帧数为\t" << n_detect_fail << std::endl <<
		"检出率为\t" << n_detect_suc / (double(std::min(374,int(g_video_cap.get(cv::CAP_PROP_POS_FRAMES)))) - n_detect_start) << std::endl <<
		"最大检测失败连续帧数为\t" << n_detect_fail_interval_Max << std::endl;
	std::cout << "检出队列中共有" << q_Circles.size() << "个有效帧" << std::endl;
	return 0;
}

// 播放进度条回调函数
void onTrackBarSlide(int pos, void*)
{
	g_video_cap.set(cv::CAP_PROP_POS_FRAMES, pos);

	if (!g_slider_dontset)
		g_slider_run = 1;
	g_slider_dontset = 0;
}

void ProSimple(const cv::Mat& srcImg)
{
	cv::Mat dstImg;
	cv::Mat midImg;
	cv::cvtColor(srcImg, midImg, cv::COLOR_BGR2GRAY);
	
	//cv::pyrDown(midImg, midImg);
	 //cv::pyrDown(midImg, midImg);


	// 形态学操作
	 //获取自定义核
	 //int nStructElementSize = 3; // 结构元素（内核矩阵）的尺寸
	 //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
		// cv::Size(2 * nStructElementSize + 1, 2 * nStructElementSize + 1));
	 //cv::morphologyEx(midImg, midImg, cv::MORPH_CLOSE, element);

	 // 漫水填充
	 //cv::Rect ccomp;
	 //cv::floodFill(midImg, cv::Point(50, 300), cv::Scalar(1), &ccomp, 
		// cv::Scalar(2), cv::Scalar(2));

	
	//// canny边缘检测
	//cv::Canny(midImg, dstImg, 50, 100, 3, true);
	//cv::imshow(WINDOW_2, dstImg);

	//imgHoughLinesP(midImg);
	 //imgHoughCircles(midImg);


	// writer << out;


	

}

// 5种基本滤波方法
cv::Mat img_filter(int opt, const cv::Mat& src_img)
{
	cv::Mat dst_img;
	switch (opt)
	{
	case 1: 	
		cv::boxFilter(src_img, dst_img, -1, cv::Size(5, 5));
		return dst_img;// 方框滤波
	case 2:
		cv::blur(src_img, dst_img, cv::Size(7, 7)); // 均值滤波
		return dst_img;
	case 3:
		cv::GaussianBlur(src_img, dst_img, cv::Size(5, 5), 3, 3); // 高斯滤波
		return dst_img;
	case 4:
		cv::medianBlur(src_img, dst_img, 3); // 中值滤波
		return dst_img;
	case 5:
		cv::bilateralFilter(src_img, dst_img, 25, int(25 * 2), 25 / 2); //双边滤波
		return dst_img;
	default:
		std::cout << "No such filter!";
	}
}
 // 用滑动条控制的5种基本滤波方法
static void on_Filter(int, void*)
{
	
	switch (g_nFilteropt)
	{
	case 1:
		cv::boxFilter(g_ROI_img, g_filtered_img, -1, cv::Size(g_nFiltervalue + 1, g_nFiltervalue + 1)); //方框滤波
		break;
	case 2:
		cv::blur(g_ROI_img, g_filtered_img, cv::Size(g_nFiltervalue + 1, g_nFiltervalue + 1)); // 均值滤波
		break;
	case 3:
		cv::GaussianBlur(g_ROI_img, g_filtered_img, cv::Size(g_nFiltervalue*2 + 1 , g_nFiltervalue*2 + 1), 3, 3); // 高斯滤波
		break;
	case 4:
		cv::medianBlur(g_ROI_img, g_filtered_img, g_nFiltervalue*2+1); // 中值滤波
		break;
	case 5:
		cv::bilateralFilter(g_ROI_img, g_filtered_img, g_nFiltervalue, 20, 20); //双边滤波
		break;
	default:
		std::cout << "No such filter!";
	}
	cv::imshow(WINDOW_2, g_filtered_img);
}


void imgDFT(const cv::InputArray src_ROI)
{
	// 将图像转化成灰度模式
	cv::Mat img_out;
	cv::cvtColor(src_ROI, img_out, cv::COLOR_BGR2GRAY);

	// 将图像延扩到最佳尺寸，边界用0补充，DFT运行速度与图片尺寸关系很大
	int m = cv::getOptimalDFTSize(img_out.rows);
	int n = cv::getOptimalDFTSize(img_out.cols);
	// 将添加的像素初始化为0.
	cv::Mat padded;
	cv::copyMakeBorder(img_out, padded, 0, m - img_out.rows, 0, n - img_out.cols, 
		cv::BORDER_CONSTANT, cv::Scalar::all(0));

	// 为傅立叶变换的结果(实部和虚部)分配存储空间
	// 将planes数组组合合并成一个多通道的数组complexI
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexI;
	merge(planes, 2, complexI);

	// 进行就地离散傅里叶变换
	dft(complexI, complexI);

	// 将复数转换为幅值，即=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes); // 将多通道数组complexI分离成几个单通道数组，planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude  
	cv::Mat magnitudeImage = planes[0];

	// 进行对数尺度(logarithmic scale)缩放
	magnitudeImage += cv::Scalar::all(1);
	log(magnitudeImage, magnitudeImage);//求自然对数

	// 剪切和重分布幅度图象限
	//若有奇数行或奇数列，进行频谱裁剪， -2补码为11111110,适合用于奇数裁成偶数
	magnitudeImage = magnitudeImage(cv::Rect(0, 0, 
		magnitudeImage.cols & -2, magnitudeImage.rows & -2)); 
	//重新排列傅立叶图像中的象限，使得原点位于图像中心  
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	cv::Mat q0(magnitudeImage, cv::Rect(0, 0, cx, cy));   // ROI区域的左上
	cv::Mat q1(magnitudeImage, cv::Rect(cx, 0, cx, cy));  // ROI区域的右上
	cv::Mat q2(magnitudeImage, cv::Rect(0, cy, cx, cy));  // ROI区域的左下
	cv::Mat q3(magnitudeImage, cv::Rect(cx, cy, cx, cy)); // ROI区域的右下
	//交换象限（左上与右下进行交换）
	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	//交换象限（右上与左下进行交换）
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// 归一化，用0到1之间的浮点值将矩阵变换为可视的图像格式
	//此句代码的OpenCV2版为：
	//normalize(magnitudeImage, magnitudeImage, 0, 1, CV_MINMAX); 
	//此句代码的OpenCV3版为:
	normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);

	// 显示效果图
	imshow(WINDOW_2, magnitudeImage);
}

// 霍夫线检测，输入边缘检测后的二值图像
void imgHoughLines( const cv::InputOutputArray srcImg )
{
	cv::Mat dstImg;
	std::vector<cv::Vec2f> lines; // 定义一个矢量结构lines存放得到的线段矢量集合
	cv::cvtColor(srcImg, dstImg, CV_GRAY2BGR);
	cv::HoughLines(srcImg, lines, 1, CV_PI / 180, 100, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		//此句代码的OpenCV2版为:
		//line( srcImage, pt1, pt2, Scalar(55,100,195), 1, CV_AA);
		//此句代码的OpenCV3版为:
		line(dstImg, pt1, pt2, cv::Scalar(55, 100, 195), 1, cv::LINE_AA);

	}
	cv::imshow(WINDOW_3, dstImg);

}

// 输入8位单通道二进制图像
void imgHoughLinesP(const cv::InputOutputArray srcImg)
{
	cv::Mat dstImg;
	std::vector<cv::Vec4i> lines; // 定义一个矢量结构lines存放得到的线段矢量集合
	cv::cvtColor(srcImg, dstImg, CV_GRAY2BGR); // 转化边缘检测后的图为彩色图
	cv::HoughLinesP(srcImg, lines, 1, CV_PI / 180, 80, 50, 10);

	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];

		line(dstImg, cv::Point (l[0],l[1]), cv::Point(l[2], l[3]), cv::Scalar(186, 88, 255), 1, cv::LINE_AA);

	}
	cv::imshow(WINDOW_3, dstImg);

}

// 霍夫圆检测，输入为8位灰度单通道图像，原始
static void on_HoughCircles(int, void*)
{
	static std::vector<cv::Vec3f> circles;
	on_Filter(g_nFiltervalue, 0);

	if (n_detect_start != 0 && n_detect_suc == 0) // 记录第一个检出圆的信息
	{
		q_Circles.push_back(circles[0]);
		n_detect_suc++;
		std::cout << "第1个检出圆的位置为：" << circles[0][1] << '\t' << circles[0][2] << std::endl;
	}
	cv::HoughCircles(g_filtered_img, circles, cv::HOUGH_GRADIENT, g_HoughVar.HC_dp, 
		g_HoughVar.HC_minDist, g_HoughVar.HC_canny, g_HoughVar.HC_CTh, g_HoughVar.HC_minR, g_HoughVar.HC_maxR);
		
	int detect_flag = 0;
	if (circles.size() != 0 && n_detect_start) // 记录第一个检出圆之后检出圆的信息
	{
		for (size_t i = 0; i < circles.size(); i++)
		{
			//参数定义
			//cv::Point center(cvRound(circles[i][0]) + g_ROI_rec.x, cvRound(circles[i][1]) + g_ROI_rec.y); 
			// ！注意ROI和源图中同一个点的对应位置关系，之后还要改一下，因为检出圆和对应的g_ROI_rec值可能不同
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]) );
			cv::Point pre_center(cvRound(q_Circles[n_detect_suc-1][0]) , cvRound(q_Circles[n_detect_suc - 1][1]) );
			

			if (dis_between_points(center, pre_center) < CIR_DIS_TH)// 求当前圆和上一个检测圆之间的距离
			{
				q_Circles.push_back(circles[i]);
				n_detect_suc++;
				std::cout << "第"<<n_detect_suc<<"个检出圆的位置为：" << circles[0][1] << '\t' << circles[0][2] << std::endl;


				if (n_detect_fail_interval > n_detect_fail_interval_Max)
					n_detect_fail_interval_Max = n_detect_fail_interval;

				n_detect_fail_interval = 0;
				detect_flag = 1;
				break;
			}
		}
	}

	if (detect_flag == 0) // 这一帧中没有合格检出圆时
	{
		n_detect_fail++;
		n_detect_fail_interval++;
	}

	for (size_t i = 0; i < circles.size(); i++)
	{
		//参数定义
		cv::Point center(cvRound(circles[i][0]) , cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		//绘制圆心
		circle(g_ROI_img, center, 3, cv::Scalar(255, 255, 0), -1, 8, 0);
		//绘制圆轮廓
		circle(g_ROI_img, center, radius, cv::Scalar(255, 50, 255), 3, 8, 0);
	}

	cv::imshow(WINDOW_3, g_ROI_img);
}

// 霍夫圆检测，输入为8位灰度单通道图像，改进后
static void on_HoughCircles1(int, void*)
{
	static std::vector<cv::Vec3f> circles;
	on_Filter(g_nFiltervalue, 0);
	static cv::Point center;
	static cv::Point pre_center;
	float radius;
	// 加一个轮廓提取的算法，存入q_Contours队列中，并在WINDOW_4窗口中画出，保存为视频数据
	std::vector<std::vector<cv::Point>> ROI_rec_contours; // ROI_rec 中的轮廓

	if (n_detect_start != 0 && n_detect_suc == 0) // 记录第一个检出圆的信息
	{
		q_Circles.push_back(circles[0]);
		n_detect_suc++;
		center.x = circles[0][0];
		center.y = circles[0][1];
		radius = circles[0][2];
		pre_center = center;
		std::cout << "第1个检出圆的位置为：" << center.x << '\t' << center.y << '\t' <<
			"半径为：" << radius << std::endl;

		// 更新ROI_rec
		g_ROI_rec.x = std::max(float(0),center.x - radius - REC_CIR_D); // 考虑到ROI_rec 左上点可能为0的情况
		g_ROI_rec.y = std::max(float(0),center.y - radius - REC_CIR_D);
		g_ROI_rec.width = g_ROI_rec.height = 2 * (radius + REC_CIR_D);
		rectangle(g_ROI_img, g_ROI_rec, cv::Scalar(255, 0, 0), 1, 8, 0);

		on_ExtractContours(0,0);
		
		circles.clear();
	}

	cv::HoughCircles(g_filtered_img(g_ROI_rec), circles, cv::HOUGH_GRADIENT, g_HoughVar.HC_dp,
		g_HoughVar.HC_minDist, g_HoughVar.HC_canny, g_HoughVar.HC_CTh, g_HoughVar.HC_minR, g_HoughVar.HC_maxR); 
	


	int detect_flag = 0; // 是否检出的标志

	if (n_detect_start && circles.size() != 0 ) // 记录第一个检出圆之后的检出圆的信息
	{
		// 判断ROI_rec中是否有合格检出圆
		pre_center = center;

		for (size_t i = 0; i < circles.size(); i++)
		{
			//参数定义，注意坐标统一到g_ROI_img中去
			center.x = cvRound(circles[i][0] + g_ROI_rec.x );
			center.y = cvRound(circles[i][1] + g_ROI_rec.y);
			radius = circles[i][2];

			// 若当前圆和上一个检出圆之间的距离在可接受范围CIR_DIS_TH内，则认为该圆为合格检出圆，存入栈中，并更新下一个ROI_rec
			if (dis_between_points(center, pre_center) < CIR_DIS_TH)
			{
				detect_flag = 1;
				// 更新最大连续未检出帧数
				if (n_detect_fail_interval > n_detect_fail_interval_Max)
					n_detect_fail_interval_Max = n_detect_fail_interval;
				n_detect_fail_interval = 0;

				// 将当前检出圆存入栈中
				q_Circles.push_back(circles[i]);
				n_detect_suc++;
				std::cout << "第" << n_detect_suc << "个检出圆的位置为：" << center.x << '\t' << center.y <<
					"\t半径为：" << radius << std::endl;

				// 更新g_ROI_rec
				g_ROI_rec.x = std::max(float(0), center.x - radius - REC_CIR_D);
				g_ROI_rec.y = std::max(float(0), center.y - radius - REC_CIR_D);
				g_ROI_rec.height = 2 * (radius + REC_CIR_D);
				g_ROI_rec.width = 2 * (radius + REC_CIR_D);
				rectangle(g_ROI_img, g_ROI_rec, cv::Scalar(255, 0, 0), 1, 8, 0);

				on_ExtractContours(0, 0);

				//// 将圆绘制出来
				//绘制圆心
				circle(g_ROI_img, center, 3, cv::Scalar(255, 255, 0), -1, 8, 0);
				////绘制圆轮廓
				//circle(g_ROI_img, center, radius, cv::Scalar(255, 50, 255), 3, 8, 0);

				break;
			}
		}

		circles.clear();

	}

	if (n_detect_start && !detect_flag) // 这一帧中没有合格检出圆时
	{
		n_detect_fail++;
		n_detect_fail_interval++;
	}

	if (n_detect_start == 0) // 未检出前将所有检测到的圆绘制出来
	{
		for (size_t i = 0; i < circles.size(); i++)
		{
			//参数定义
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			//绘制圆心
			circle(g_ROI_img, center, 3, cv::Scalar(255, 255, 0), -1, 8, 0);
			//绘制圆轮廓
			circle(g_ROI_img, center, radius, cv::Scalar(255, 50, 255), 3, 8, 0);

		}
	}
	cv::imshow(WINDOW_3, g_ROI_img);
}

// 霍夫圆检测，输入为8位灰度单通道图像
void imgHoughCircles()
{
	std::vector<cv::Vec3f> circles;
	on_Filter(g_nFiltervalue, 0);

	cv::HoughCircles(g_filtered_img, circles, cv::HOUGH_GRADIENT, 1,
		1000, 50, 30, 40, 70);

	for (size_t i = 0; i < circles.size(); i++)
	{
		//参数定义
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		//绘制圆心
		circle(g_ROI_img, center, 3, cv::Scalar(255, 255, 0), -1, 8, 0);
		//绘制圆轮廓
		circle(g_ROI_img, center, radius, cv::Scalar(255, 50, 255), 3, 8, 0);
	}

	cv::imshow(WINDOW_3, g_ROI_img);
}

static void on_MouseHandle(int event, int x, int y, int flags, void*)
{
	switch (event)
	{
		//鼠标移动消息
	case cv::EVENT_MOUSEMOVE:
	{
		if (g_bDrawingBox)//如果是否进行绘制的标识符为真，则记录下长和宽到RECT型变量中
		{
			g_ROI_rec.width = x - g_ROI_rec.x;
			g_ROI_rec.height = y - g_ROI_rec.y;

		}
	}
	break;

	//左键按下消息
	case cv::EVENT_LBUTTONDOWN:
	{
		g_bDrawingBox = true;
		g_slider_run = 0;
		g_ROI_rec = cv::Rect(x, y, 0, 0);//记录起始点
		n_detect_start = g_video_cap.get(cv::CAP_PROP_POS_FRAMES);
		std::cout << "第一次检出圆的帧位为" << n_detect_start << std::endl;
		std::cout << g_ROI_rec.x << '\t' << g_ROI_rec.y << std::endl;
	}
	break;

	//左键抬起消息
	case cv::EVENT_LBUTTONUP:
	{
		g_bDrawingBox = false;//置标识符为false
		//对宽和高小于0的处理
		if (g_ROI_rec.width < 0)
		{
			g_ROI_rec.x += g_ROI_rec.width;
			g_ROI_rec.width *= -1;
		}

		if (g_ROI_rec.height < 0)
		{
			g_ROI_rec.y += g_ROI_rec.height;
			g_ROI_rec.height *= -1;
		}
		//调用函数进行绘制
		g_slider_run = -1;
		std::cout << g_ROI_rec.x+ g_ROI_rec.width << '\t' << g_ROI_rec.y+g_ROI_rec.height << std::endl;

		rectangle(g_ROI_img, g_ROI_rec,cv::Scalar(255,0,0),1,8,0);
		g_ROI_rec_img = g_ROI_img(g_ROI_rec);
		//cv::imshow(WINDOW_3, g_ROI_img);
		
	}
	break;

	}
}

static void on_MouseHandle1(int event, int x, int y, int flags, void*)
{
	switch (event)
	{
		//鼠标移动消息
	case cv::EVENT_MOUSEMOVE:

	break;

	//左键按下消息
	case cv::EVENT_LBUTTONDOWN:
	{
		g_bDrawingBox = true;
		g_slider_run = 0; // 暂停播放
		n_detect_start = g_video_cap.get(cv::CAP_PROP_POS_FRAMES);
		std::cout << "第一次检出圆的帧位为" << n_detect_start << std::endl;
	}
	break;

	//左键抬起消息
	case cv::EVENT_LBUTTONUP:
	{
		g_bDrawingBox = false;//置标识符为false
		
		g_slider_run = -1;

		// g_ROI_rec_img = g_ROI_img(g_ROI_rec);
		//cv::imshow(WINDOW_3, g_ROI_img);
	}
	break;

	}
}

double dis_between_points(cv::Point& A, cv::Point& B)
{
	return sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
}

void on_ExtractContours(int, void*)
{
	std::vector<std::vector<cv::Point>> ROI_rec_contours; // ROI_rec 中的轮廓

	// cv::Canny(g_filtered_img(g_ROI_rec), g_filtered_ROI_rec_binary_img, g_HoughVar.HC_canny, g_HoughVar.HC_canny * 2, 3); // 先进行canny检测变成二值图像

	 cv::threshold(g_filtered_img(g_ROI_rec), g_filtered_ROI_rec_binary_img, g_ROI_rec_nThresholdValue, 255, g_ROI_rec_nThresholdType);

	//cv::adaptiveThreshold(g_filtered_img(g_ROI_rec), g_filtered_ROI_rec_binary_img, 255,
	//	g_ROI_rec_naThresholdMethod,
	//	g_ROI_rec_nThresholdType,
	//	2*g_ROI_rec_naThresholdBlocksize+1,
	//	g_ROI_rec_naThresholdC);

	findContours(g_filtered_ROI_rec_binary_img,
		ROI_rec_contours, // 轮廓数组
		CV_RETR_EXTERNAL, // 获取外轮廓
		CV_CHAIN_APPROX_NONE);// 获取每个轮廓上每个像素点的位置

	cv::Mat ROI_rec_contours_img(g_ROI_rec.size(), CV_8U, cv::Scalar(255));
	drawContours(ROI_rec_contours_img, ROI_rec_contours,
		-1,//绘制所有轮廓
		cv::Scalar(0), // 绘制轮廓为黑色
		2, // 轮廓线宽为3
		8, // 线型
		cv::Mat(), // 可选层次结构信息
		INT_MAX, // 用于绘制轮廓的最大等级
		cv::Point()); // ？offset的使用 -g_ROI_rec.x, -g_ROI_rec.y

	cv::imshow(WINDOW_4, ROI_rec_contours_img);
	
}

