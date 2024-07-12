#ifndef FACEREGCONITION_H
#define FACEREGCONITION_H
#include "enginebaseregconize.h"

class FaceRegconition : public EngineBase2
{
public:
    FaceRegconition();
    void InitEngine2() override;
    std::vector<float> InferenceEngine2(cv::Mat &image) override;
    cv::Size modelShape = cv::Size(112,112);
    float CousineDistance(std::vector<float> v1, std::vector<float> v2);
private:
    cv::Mat PreprocessImage(cv::Mat &image) override;
private:
    Ort::RunOptions runOption;
};

#endif // FACEREGCONITION_H
