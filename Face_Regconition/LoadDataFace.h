#ifndef LOADDATAFACE_H
#define LOADDATAFACE_H
#include <QTextStream>
#include <QFile>
class LoadDataFace
{
public:
    LoadDataFace();
    void SaveUserData(const std::string name, const std::vector<float> faceData);
    std::unordered_map<std::string, std::vector<float>> LoadUserData();
    float CalculateSimilarity(const std::vector<float> data1, const std::vector<float> data2);
};

#endif // LOADDATAFACE_H
