#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

// ȫ�ֱ�������
const std::string VIDEO_SRC = "./detect_carotid_artery/230213_1.mp4"; 
const std::string VIDEO_OUT = "./detect_carotid_artery/230213_1_out.avi";
const std::string WINDOW_1 = "origin_video"; // ԭ��Ƶ
const std::string WINDOW_2 = "process_filter"; // �˲�����Ƶ
const std::string WINDOW_3 = "process_detect"; // ������Ƶ
const std::string WINDOW_4 = "process_extract"; // ������ȡ�� ROI_rec���� ��Ƶ

cv::VideoCapture g_video_cap; 
cv::VideoWriter g_video_writer;


int g_slider_position = 65; // ��Ƶ��ʼ���ŵ�λ������
int g_slider_run = 1; //<0: run mode; 1: single mode; 0: stop mode
int g_slider_dontset = 0;

cv::Mat g_src_img; // Դͼ
cv::Mat	g_ROI_img; // 600X800 �Ҷ�ͼ
cv::Mat g_filtered_img; // g_ROI_img �˲���ͼ��
cv::Mat	g_ROI_rec_img; // 600X800ROI �е� ROI_rec����̽������
cv::Mat g_filtered_ROI_rec_binary_img; // ROI_rec������ͼ�� ��ֵ�����

int g_nFilteropt = 3; // �˲�����
int g_nFiltervalue = 1; // �˲���������Ϊ���򡢾�ֵ����˹�˲�ʱ����ʾ�˵Ĵ�С

int g_ROI_rec_nThresholdValue = 100; // ��ֵ������ֵ
int g_ROI_rec_nThresholdType = 0;	// ��ֵ����ֵ���� 0��CV_THRESH_BINARY

int g_ROI_rec_naThresholdMethod = CV_ADAPTIVE_THRESH_MEAN_C; // ����Ӧ��ֵ�㷨
int g_ROI_rec_naThresholdBlocksize = 3; // ���ڼ�����ֵ������ߴ�
int g_ROI_rec_naThresholdC = 0; // ����Ӧ��ֵ�ĳ�������֪�������




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
const int CIR_DIS_TH = 20; // ����һ���Բ�͵�ǰ���ԲԲ�ľ��볬���˼���ֵʱ����Ϊ���ʧ��

// ���ص�����ȫ�ֱ�������
cv::Rect g_ROI_rec = cv::Rect(0, 0, ROI_WIDTH, ROI_HIGHT); // 600X800 �еľ������򣬳�ʼ��Ϊ��������
bool g_bDrawingBox = false; // �Ƿ���л���

int n_detect_start = 0; // ����ĵ�һ֡��֡λ
int n_detect_suc = 0; // �� n_detect_start ��ʼ�ļ��֡��
int n_detect_fail = 0; // �� n_detect_start ��ʼ��δ���֡��
int n_detect_fail_interval = 0; // ����δ���֡��
int n_detect_fail_interval_Max = 0; //����δ���֡�������ֵ
std::vector<cv::Vec3f> q_Circles; // ��źϸ���Բ
std::vector<std::vector<std::vector<cv::Point>>> q_Contours; // ��źϸ���Բȷ���� ROI_rec �м���������
const int REC_CIR_D = 20; // ��Ϊ�趨�� ROI_rec ���� �ϸ���Բ �ı߽�����������
const int N_COUNT_FAIL_INTERVAL = 100; // ��Ϊ�趨�� �������δ���֡�� ��ֵ��������������ΪĿ�����ʧ�ܣ��������




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

void on_ExtractContours(int, void*); // ����ֵ�ָ�ص�����


int main()
{

	//������Ƶ
	// cv::namedWindow(WINDOW_1, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_2, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_3, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_4, CV_WINDOW_AUTOSIZE);
	g_video_cap.open(VIDEO_SRC);
	if (!g_video_cap.isOpened()) return -1;
	
	// ��Ƶ������Ϣ
	int frames = (int)g_video_cap.get(cv::CAP_PROP_FRAME_COUNT);
	int tmpw = (int)g_video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int tmph = (int)g_video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Video has " << frames << " frames of dimensions("
		<< tmpw << ", " << tmph << ")." << std::endl;
	g_video_cap.set(cv::CAP_PROP_POS_FRAMES, g_slider_position);

	//�����������
	double fps = g_video_cap.get(cv::CAP_PROP_FPS);
	cv::Size size_ROI(((ROI_WIDTH+1)/2+1)/2, ((ROI_HIGHT + 1) / 2 + 1) / 2);
	g_video_writer.open(VIDEO_OUT, CV_FOURCC('M', 'J', 'P', 'G'), fps, size_ROI,false);

	//����������
	cv::createTrackbar("Position", WINDOW_2, &g_slider_position, frames,
		onTrackBarSlide); // ���Ż�����
	cv::createTrackbar("KernelSize", WINDOW_2, &g_nFiltervalue, 40, on_Filter); // �˲�������
	cv::createTrackbar("FilterType", WINDOW_2, &g_nFilteropt, 5, on_Filter); // �˲�����
	cv::createTrackbar("b_Th_Value", WINDOW_2, &g_ROI_rec_nThresholdValue, 255, on_ExtractContours); 
	cv::createTrackbar("b_Th_Type", WINDOW_2, &g_ROI_rec_nThresholdType, 4, on_ExtractContours);
	cv::createTrackbar("b_aTh_Meth", WINDOW_2, &g_ROI_rec_naThresholdMethod, 1, on_ExtractContours);
	cv::createTrackbar("b_aTh_blockSize", WINDOW_2, &g_ROI_rec_naThresholdBlocksize, 7, on_ExtractContours);
	cv::createTrackbar("b_aTh_C", WINDOW_2, &g_ROI_rec_naThresholdC, 255, on_ExtractContours);

	// ����Բ�����ز���������
	cv::createTrackbar("HC_dp", WINDOW_3, &(g_HoughVar.HC_dp), 5, on_HoughCircles); // �ۼ���ͼ��ķֱ���
	cv::createTrackbar("HC_minDist", WINDOW_3, &(g_HoughVar.HC_minDist), 1000, on_HoughCircles); //�ɼ�����СԲ����
	cv::createTrackbar("HC_canny", WINDOW_3, &(g_HoughVar.HC_canny), 200, on_HoughCircles); // canny��Ե������ֵ
	cv::createTrackbar("HC_cirTh", WINDOW_3, &(g_HoughVar.HC_CTh), 60, on_HoughCircles); // Բ���ۼ�����ֵ
	cv::createTrackbar("HC_minR", WINDOW_3, &(g_HoughVar.HC_minR), 100, on_HoughCircles); // �ɼ�����СԲ�뾶
	cv::createTrackbar("HC_maxR", WINDOW_3, &(g_HoughVar.HC_maxR), 100, on_HoughCircles); // �ɼ������Բ�뾶


	
	//�������Ȥ������Χ 
	cv::Mat src_ROI;
	cv::Rect rect(250, 110, ROI_WIDTH, ROI_HIGHT);
	cv::setMouseCallback(WINDOW_3, on_MouseHandle1); // ע�����ص�����

	while (1)
	{
		if (g_slider_run != 0) {
			// ��ȡһ֡ͼ��
			g_video_cap >> g_src_img; if (!g_src_img.data) break;
			src_ROI = g_src_img(rect);
			cvtColor(src_ROI, g_ROI_img, cv::COLOR_BGR2GRAY); // ת�ɻҶ�ͼ

			//���»�������λ��
			int current_pos = (int)g_video_cap.get(cv::CAP_PROP_POS_FRAMES);
			g_slider_dontset = 1;
			cv::setTrackbarPos("Position", WINDOW_2, current_pos);

			////��ʾԭʼ����Ȥ����
			//cv::imshow(WINDOW_1, src_ROI);
			

			// ����Բ���
			//imgHoughCircles();
			if (n_detect_fail_interval > N_COUNT_FAIL_INTERVAL)
			{
				std::cout << "��" << int(g_video_cap.get(cv::CAP_PROP_POS_FRAMES)) << "ʱĿ�����ʧ�ܣ�" << std::endl <<
					"�����" << int(g_video_cap.get(cv::CAP_PROP_POS_FRAMES)) - n_detect_start << "֡" << std::endl <<
					"�ϸ���ԲΪ\t" << n_detect_suc << std::endl <<
					"δ���֡��Ϊ\t" << n_detect_fail << std::endl <<
					"�����Ϊ\t" << n_detect_suc / (double(g_video_cap.get(cv::CAP_PROP_POS_FRAMES)) - n_detect_start) << std::endl <<
					"�����ʧ������֡��Ϊ\t" << n_detect_fail_interval_Max << std::endl;
				return -1;
			}
			on_HoughCircles1(0, 0);
			 

			g_slider_run -= 1;
		}

		// һЩ���̿��ƵĲ�������
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
		"�����" << std::min(374,int(g_video_cap.get(cv::CAP_PROP_POS_FRAMES))) - n_detect_start << "֡" << std::endl <<
		"�ϸ���ԲΪ\t" << n_detect_suc << std::endl <<
		"δ���֡��Ϊ\t" << n_detect_fail << std::endl <<
		"�����Ϊ\t" << n_detect_suc / (double(std::min(374,int(g_video_cap.get(cv::CAP_PROP_POS_FRAMES)))) - n_detect_start) << std::endl <<
		"�����ʧ������֡��Ϊ\t" << n_detect_fail_interval_Max << std::endl;
	std::cout << "��������й���" << q_Circles.size() << "����Ч֡" << std::endl;
	return 0;
}

// ���Ž������ص�����
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


	// ��̬ѧ����
	 //��ȡ�Զ����
	 //int nStructElementSize = 3; // �ṹԪ�أ��ں˾��󣩵ĳߴ�
	 //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
		// cv::Size(2 * nStructElementSize + 1, 2 * nStructElementSize + 1));
	 //cv::morphologyEx(midImg, midImg, cv::MORPH_CLOSE, element);

	 // ��ˮ���
	 //cv::Rect ccomp;
	 //cv::floodFill(midImg, cv::Point(50, 300), cv::Scalar(1), &ccomp, 
		// cv::Scalar(2), cv::Scalar(2));

	
	//// canny��Ե���
	//cv::Canny(midImg, dstImg, 50, 100, 3, true);
	//cv::imshow(WINDOW_2, dstImg);

	//imgHoughLinesP(midImg);
	 //imgHoughCircles(midImg);


	// writer << out;


	

}

// 5�ֻ����˲�����
cv::Mat img_filter(int opt, const cv::Mat& src_img)
{
	cv::Mat dst_img;
	switch (opt)
	{
	case 1: 	
		cv::boxFilter(src_img, dst_img, -1, cv::Size(5, 5));
		return dst_img;// �����˲�
	case 2:
		cv::blur(src_img, dst_img, cv::Size(7, 7)); // ��ֵ�˲�
		return dst_img;
	case 3:
		cv::GaussianBlur(src_img, dst_img, cv::Size(5, 5), 3, 3); // ��˹�˲�
		return dst_img;
	case 4:
		cv::medianBlur(src_img, dst_img, 3); // ��ֵ�˲�
		return dst_img;
	case 5:
		cv::bilateralFilter(src_img, dst_img, 25, int(25 * 2), 25 / 2); //˫���˲�
		return dst_img;
	default:
		std::cout << "No such filter!";
	}
}
 // �û��������Ƶ�5�ֻ����˲�����
static void on_Filter(int, void*)
{
	
	switch (g_nFilteropt)
	{
	case 1:
		cv::boxFilter(g_ROI_img, g_filtered_img, -1, cv::Size(g_nFiltervalue + 1, g_nFiltervalue + 1)); //�����˲�
		break;
	case 2:
		cv::blur(g_ROI_img, g_filtered_img, cv::Size(g_nFiltervalue + 1, g_nFiltervalue + 1)); // ��ֵ�˲�
		break;
	case 3:
		cv::GaussianBlur(g_ROI_img, g_filtered_img, cv::Size(g_nFiltervalue*2 + 1 , g_nFiltervalue*2 + 1), 3, 3); // ��˹�˲�
		break;
	case 4:
		cv::medianBlur(g_ROI_img, g_filtered_img, g_nFiltervalue*2+1); // ��ֵ�˲�
		break;
	case 5:
		cv::bilateralFilter(g_ROI_img, g_filtered_img, g_nFiltervalue, 20, 20); //˫���˲�
		break;
	default:
		std::cout << "No such filter!";
	}
	cv::imshow(WINDOW_2, g_filtered_img);
}


void imgDFT(const cv::InputArray src_ROI)
{
	// ��ͼ��ת���ɻҶ�ģʽ
	cv::Mat img_out;
	cv::cvtColor(src_ROI, img_out, cv::COLOR_BGR2GRAY);

	// ��ͼ����������ѳߴ磬�߽���0���䣬DFT�����ٶ���ͼƬ�ߴ��ϵ�ܴ�
	int m = cv::getOptimalDFTSize(img_out.rows);
	int n = cv::getOptimalDFTSize(img_out.cols);
	// ����ӵ����س�ʼ��Ϊ0.
	cv::Mat padded;
	cv::copyMakeBorder(img_out, padded, 0, m - img_out.rows, 0, n - img_out.cols, 
		cv::BORDER_CONSTANT, cv::Scalar::all(0));

	// Ϊ����Ҷ�任�Ľ��(ʵ�����鲿)����洢�ռ�
	// ��planes������Ϻϲ���һ����ͨ��������complexI
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexI;
	merge(planes, 2, complexI);

	// ���о͵���ɢ����Ҷ�任
	dft(complexI, complexI);

	// ������ת��Ϊ��ֵ����=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes); // ����ͨ������complexI����ɼ�����ͨ�����飬planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude  
	cv::Mat magnitudeImage = planes[0];

	// ���ж����߶�(logarithmic scale)����
	magnitudeImage += cv::Scalar::all(1);
	log(magnitudeImage, magnitudeImage);//����Ȼ����

	// ���к��طֲ�����ͼ����
	//���������л������У�����Ƶ�ײü��� -2����Ϊ11111110,�ʺ����������ó�ż��
	magnitudeImage = magnitudeImage(cv::Rect(0, 0, 
		magnitudeImage.cols & -2, magnitudeImage.rows & -2)); 
	//�������и���Ҷͼ���е����ޣ�ʹ��ԭ��λ��ͼ������  
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	cv::Mat q0(magnitudeImage, cv::Rect(0, 0, cx, cy));   // ROI���������
	cv::Mat q1(magnitudeImage, cv::Rect(cx, 0, cx, cy));  // ROI���������
	cv::Mat q2(magnitudeImage, cv::Rect(0, cy, cx, cy));  // ROI���������
	cv::Mat q3(magnitudeImage, cv::Rect(cx, cy, cx, cy)); // ROI���������
	//�������ޣ����������½��н�����
	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	//�������ޣ����������½��н�����
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// ��һ������0��1֮��ĸ���ֵ������任Ϊ���ӵ�ͼ���ʽ
	//�˾�����OpenCV2��Ϊ��
	//normalize(magnitudeImage, magnitudeImage, 0, 1, CV_MINMAX); 
	//�˾�����OpenCV3��Ϊ:
	normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);

	// ��ʾЧ��ͼ
	imshow(WINDOW_2, magnitudeImage);
}

// �����߼�⣬�����Ե����Ķ�ֵͼ��
void imgHoughLines( const cv::InputOutputArray srcImg )
{
	cv::Mat dstImg;
	std::vector<cv::Vec2f> lines; // ����һ��ʸ���ṹlines��ŵõ����߶�ʸ������
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
		//�˾�����OpenCV2��Ϊ:
		//line( srcImage, pt1, pt2, Scalar(55,100,195), 1, CV_AA);
		//�˾�����OpenCV3��Ϊ:
		line(dstImg, pt1, pt2, cv::Scalar(55, 100, 195), 1, cv::LINE_AA);

	}
	cv::imshow(WINDOW_3, dstImg);

}

// ����8λ��ͨ��������ͼ��
void imgHoughLinesP(const cv::InputOutputArray srcImg)
{
	cv::Mat dstImg;
	std::vector<cv::Vec4i> lines; // ����һ��ʸ���ṹlines��ŵõ����߶�ʸ������
	cv::cvtColor(srcImg, dstImg, CV_GRAY2BGR); // ת����Ե�����ͼΪ��ɫͼ
	cv::HoughLinesP(srcImg, lines, 1, CV_PI / 180, 80, 50, 10);

	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];

		line(dstImg, cv::Point (l[0],l[1]), cv::Point(l[2], l[3]), cv::Scalar(186, 88, 255), 1, cv::LINE_AA);

	}
	cv::imshow(WINDOW_3, dstImg);

}

// ����Բ��⣬����Ϊ8λ�Ҷȵ�ͨ��ͼ��ԭʼ
static void on_HoughCircles(int, void*)
{
	static std::vector<cv::Vec3f> circles;
	on_Filter(g_nFiltervalue, 0);

	if (n_detect_start != 0 && n_detect_suc == 0) // ��¼��һ�����Բ����Ϣ
	{
		q_Circles.push_back(circles[0]);
		n_detect_suc++;
		std::cout << "��1�����Բ��λ��Ϊ��" << circles[0][1] << '\t' << circles[0][2] << std::endl;
	}
	cv::HoughCircles(g_filtered_img, circles, cv::HOUGH_GRADIENT, g_HoughVar.HC_dp, 
		g_HoughVar.HC_minDist, g_HoughVar.HC_canny, g_HoughVar.HC_CTh, g_HoughVar.HC_minR, g_HoughVar.HC_maxR);
		
	int detect_flag = 0;
	if (circles.size() != 0 && n_detect_start) // ��¼��һ�����Բ֮����Բ����Ϣ
	{
		for (size_t i = 0; i < circles.size(); i++)
		{
			//��������
			//cv::Point center(cvRound(circles[i][0]) + g_ROI_rec.x, cvRound(circles[i][1]) + g_ROI_rec.y); 
			// ��ע��ROI��Դͼ��ͬһ����Ķ�Ӧλ�ù�ϵ��֮��Ҫ��һ�£���Ϊ���Բ�Ͷ�Ӧ��g_ROI_recֵ���ܲ�ͬ
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]) );
			cv::Point pre_center(cvRound(q_Circles[n_detect_suc-1][0]) , cvRound(q_Circles[n_detect_suc - 1][1]) );
			

			if (dis_between_points(center, pre_center) < CIR_DIS_TH)// ��ǰԲ����һ�����Բ֮��ľ���
			{
				q_Circles.push_back(circles[i]);
				n_detect_suc++;
				std::cout << "��"<<n_detect_suc<<"�����Բ��λ��Ϊ��" << circles[0][1] << '\t' << circles[0][2] << std::endl;


				if (n_detect_fail_interval > n_detect_fail_interval_Max)
					n_detect_fail_interval_Max = n_detect_fail_interval;

				n_detect_fail_interval = 0;
				detect_flag = 1;
				break;
			}
		}
	}

	if (detect_flag == 0) // ��һ֡��û�кϸ���Բʱ
	{
		n_detect_fail++;
		n_detect_fail_interval++;
	}

	for (size_t i = 0; i < circles.size(); i++)
	{
		//��������
		cv::Point center(cvRound(circles[i][0]) , cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		//����Բ��
		circle(g_ROI_img, center, 3, cv::Scalar(255, 255, 0), -1, 8, 0);
		//����Բ����
		circle(g_ROI_img, center, radius, cv::Scalar(255, 50, 255), 3, 8, 0);
	}

	cv::imshow(WINDOW_3, g_ROI_img);
}

// ����Բ��⣬����Ϊ8λ�Ҷȵ�ͨ��ͼ�񣬸Ľ���
static void on_HoughCircles1(int, void*)
{
	static std::vector<cv::Vec3f> circles;
	on_Filter(g_nFiltervalue, 0);
	static cv::Point center;
	static cv::Point pre_center;
	float radius;
	// ��һ��������ȡ���㷨������q_Contours�����У�����WINDOW_4�����л���������Ϊ��Ƶ����
	std::vector<std::vector<cv::Point>> ROI_rec_contours; // ROI_rec �е�����

	if (n_detect_start != 0 && n_detect_suc == 0) // ��¼��һ�����Բ����Ϣ
	{
		q_Circles.push_back(circles[0]);
		n_detect_suc++;
		center.x = circles[0][0];
		center.y = circles[0][1];
		radius = circles[0][2];
		pre_center = center;
		std::cout << "��1�����Բ��λ��Ϊ��" << center.x << '\t' << center.y << '\t' <<
			"�뾶Ϊ��" << radius << std::endl;

		// ����ROI_rec
		g_ROI_rec.x = std::max(float(0),center.x - radius - REC_CIR_D); // ���ǵ�ROI_rec ���ϵ����Ϊ0�����
		g_ROI_rec.y = std::max(float(0),center.y - radius - REC_CIR_D);
		g_ROI_rec.width = g_ROI_rec.height = 2 * (radius + REC_CIR_D);
		rectangle(g_ROI_img, g_ROI_rec, cv::Scalar(255, 0, 0), 1, 8, 0);

		on_ExtractContours(0,0);
		
		circles.clear();
	}

	cv::HoughCircles(g_filtered_img(g_ROI_rec), circles, cv::HOUGH_GRADIENT, g_HoughVar.HC_dp,
		g_HoughVar.HC_minDist, g_HoughVar.HC_canny, g_HoughVar.HC_CTh, g_HoughVar.HC_minR, g_HoughVar.HC_maxR); 
	


	int detect_flag = 0; // �Ƿ����ı�־

	if (n_detect_start && circles.size() != 0 ) // ��¼��һ�����Բ֮��ļ��Բ����Ϣ
	{
		// �ж�ROI_rec���Ƿ��кϸ���Բ
		pre_center = center;

		for (size_t i = 0; i < circles.size(); i++)
		{
			//�������壬ע������ͳһ��g_ROI_img��ȥ
			center.x = cvRound(circles[i][0] + g_ROI_rec.x );
			center.y = cvRound(circles[i][1] + g_ROI_rec.y);
			radius = circles[i][2];

			// ����ǰԲ����һ�����Բ֮��ľ����ڿɽ��ܷ�ΧCIR_DIS_TH�ڣ�����Ϊ��ԲΪ�ϸ���Բ������ջ�У���������һ��ROI_rec
			if (dis_between_points(center, pre_center) < CIR_DIS_TH)
			{
				detect_flag = 1;
				// �����������δ���֡��
				if (n_detect_fail_interval > n_detect_fail_interval_Max)
					n_detect_fail_interval_Max = n_detect_fail_interval;
				n_detect_fail_interval = 0;

				// ����ǰ���Բ����ջ��
				q_Circles.push_back(circles[i]);
				n_detect_suc++;
				std::cout << "��" << n_detect_suc << "�����Բ��λ��Ϊ��" << center.x << '\t' << center.y <<
					"\t�뾶Ϊ��" << radius << std::endl;

				// ����g_ROI_rec
				g_ROI_rec.x = std::max(float(0), center.x - radius - REC_CIR_D);
				g_ROI_rec.y = std::max(float(0), center.y - radius - REC_CIR_D);
				g_ROI_rec.height = 2 * (radius + REC_CIR_D);
				g_ROI_rec.width = 2 * (radius + REC_CIR_D);
				rectangle(g_ROI_img, g_ROI_rec, cv::Scalar(255, 0, 0), 1, 8, 0);

				on_ExtractContours(0, 0);

				//// ��Բ���Ƴ���
				//����Բ��
				circle(g_ROI_img, center, 3, cv::Scalar(255, 255, 0), -1, 8, 0);
				////����Բ����
				//circle(g_ROI_img, center, radius, cv::Scalar(255, 50, 255), 3, 8, 0);

				break;
			}
		}

		circles.clear();

	}

	if (n_detect_start && !detect_flag) // ��һ֡��û�кϸ���Բʱ
	{
		n_detect_fail++;
		n_detect_fail_interval++;
	}

	if (n_detect_start == 0) // δ���ǰ�����м�⵽��Բ���Ƴ���
	{
		for (size_t i = 0; i < circles.size(); i++)
		{
			//��������
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			//����Բ��
			circle(g_ROI_img, center, 3, cv::Scalar(255, 255, 0), -1, 8, 0);
			//����Բ����
			circle(g_ROI_img, center, radius, cv::Scalar(255, 50, 255), 3, 8, 0);

		}
	}
	cv::imshow(WINDOW_3, g_ROI_img);
}

// ����Բ��⣬����Ϊ8λ�Ҷȵ�ͨ��ͼ��
void imgHoughCircles()
{
	std::vector<cv::Vec3f> circles;
	on_Filter(g_nFiltervalue, 0);

	cv::HoughCircles(g_filtered_img, circles, cv::HOUGH_GRADIENT, 1,
		1000, 50, 30, 40, 70);

	for (size_t i = 0; i < circles.size(); i++)
	{
		//��������
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		//����Բ��
		circle(g_ROI_img, center, 3, cv::Scalar(255, 255, 0), -1, 8, 0);
		//����Բ����
		circle(g_ROI_img, center, radius, cv::Scalar(255, 50, 255), 3, 8, 0);
	}

	cv::imshow(WINDOW_3, g_ROI_img);
}

static void on_MouseHandle(int event, int x, int y, int flags, void*)
{
	switch (event)
	{
		//����ƶ���Ϣ
	case cv::EVENT_MOUSEMOVE:
	{
		if (g_bDrawingBox)//����Ƿ���л��Ƶı�ʶ��Ϊ�棬���¼�³��Ϳ�RECT�ͱ�����
		{
			g_ROI_rec.width = x - g_ROI_rec.x;
			g_ROI_rec.height = y - g_ROI_rec.y;

		}
	}
	break;

	//���������Ϣ
	case cv::EVENT_LBUTTONDOWN:
	{
		g_bDrawingBox = true;
		g_slider_run = 0;
		g_ROI_rec = cv::Rect(x, y, 0, 0);//��¼��ʼ��
		n_detect_start = g_video_cap.get(cv::CAP_PROP_POS_FRAMES);
		std::cout << "��һ�μ��Բ��֡λΪ" << n_detect_start << std::endl;
		std::cout << g_ROI_rec.x << '\t' << g_ROI_rec.y << std::endl;
	}
	break;

	//���̧����Ϣ
	case cv::EVENT_LBUTTONUP:
	{
		g_bDrawingBox = false;//�ñ�ʶ��Ϊfalse
		//�Կ�͸�С��0�Ĵ���
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
		//���ú������л���
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
		//����ƶ���Ϣ
	case cv::EVENT_MOUSEMOVE:

	break;

	//���������Ϣ
	case cv::EVENT_LBUTTONDOWN:
	{
		g_bDrawingBox = true;
		g_slider_run = 0; // ��ͣ����
		n_detect_start = g_video_cap.get(cv::CAP_PROP_POS_FRAMES);
		std::cout << "��һ�μ��Բ��֡λΪ" << n_detect_start << std::endl;
	}
	break;

	//���̧����Ϣ
	case cv::EVENT_LBUTTONUP:
	{
		g_bDrawingBox = false;//�ñ�ʶ��Ϊfalse
		
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
	std::vector<std::vector<cv::Point>> ROI_rec_contours; // ROI_rec �е�����

	// cv::Canny(g_filtered_img(g_ROI_rec), g_filtered_ROI_rec_binary_img, g_HoughVar.HC_canny, g_HoughVar.HC_canny * 2, 3); // �Ƚ���canny����ɶ�ֵͼ��

	 cv::threshold(g_filtered_img(g_ROI_rec), g_filtered_ROI_rec_binary_img, g_ROI_rec_nThresholdValue, 255, g_ROI_rec_nThresholdType);

	//cv::adaptiveThreshold(g_filtered_img(g_ROI_rec), g_filtered_ROI_rec_binary_img, 255,
	//	g_ROI_rec_naThresholdMethod,
	//	g_ROI_rec_nThresholdType,
	//	2*g_ROI_rec_naThresholdBlocksize+1,
	//	g_ROI_rec_naThresholdC);

	findContours(g_filtered_ROI_rec_binary_img,
		ROI_rec_contours, // ��������
		CV_RETR_EXTERNAL, // ��ȡ������
		CV_CHAIN_APPROX_NONE);// ��ȡÿ��������ÿ�����ص��λ��

	cv::Mat ROI_rec_contours_img(g_ROI_rec.size(), CV_8U, cv::Scalar(255));
	drawContours(ROI_rec_contours_img, ROI_rec_contours,
		-1,//������������
		cv::Scalar(0), // ��������Ϊ��ɫ
		2, // �����߿�Ϊ3
		8, // ����
		cv::Mat(), // ��ѡ��νṹ��Ϣ
		INT_MAX, // ���ڻ������������ȼ�
		cv::Point()); // ��offset��ʹ�� -g_ROI_rec.x, -g_ROI_rec.y

	cv::imshow(WINDOW_4, ROI_rec_contours_img);
	
}

