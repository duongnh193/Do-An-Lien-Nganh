#ifndef FACEDETECT_H
#define FACEDETECT_H
#include "enginebasedetect.h"
class FaceDetect : public EngineBase
{
public:
    FaceDetect();
    // EngineBase interface
    void InitEngine() override;
    std::vector<cv::Rect> InferenceEngine(cv::Mat image);
    // void DetectResultChecker(std::vector<cv::Rect> FaceDetected, cv::Mat originImg)  override;
    cv::Mat PreVideo(cv::Mat frame, cv::VideoCapture cap);

private:

    cv::Mat PreprocessImage(cv::Mat &image) override;
    std::vector<cv::Rect> PosprocessImage(cv::Mat &image, cv::Mat &sessionResult);
    void NMS(std::vector<cv::Rect> &rects);
private:
    Ort::RunOptions runOption;
};

#endif // FACEDETECT_H
