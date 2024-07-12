#ifndef NUDEDETECTOR_H
#define NUDEDETECTOR_H
#include <enginebasedetect.h>

class NudeDetector : public  EngineBase
{
public:
    std::vector<std::string> classes = {
        "FEMALE_GENITALIA_COVERED",
        "FACE_FEMALE",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "FACE_MALE",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED"
    };

    struct NudeInfo{
        std::string className;
        cv::Rect bbox;
        int classId;
        float confident;
    };
    NudeDetector();
    void InitEngine() override;
    std::vector<float> InferenceEngine(cv::Mat &image) override;
private:
    cv::Mat PreprocessImage(cv::Mat &image) override;
    std::vector<NudeInfo> PosprocessImage(cv::Mat &image, cv::Mat &sessionResult);
    std::vector<NudeInfo> NMS(std::vector<NudeInfo> bboxes);
private:
    Ort::RunOptions runOption;
};


#endif // NUDEDETECTOR_H
