#include "LoadDataFace.h"
#include "utils.h"
LoadDataFace::LoadDataFace() {}

void LoadDataFace::SaveUserData(const std::string name, const std::vector<float> faceData) {
    QFile file("/home/duongnh/Projects/Qt/Face_Regconition/build/data.txt");

    if (file.open(QIODevice::ReadWrite | QIODevice::Text | QIODevice::Append)) {
        QTextStream stream(&file);
        QString qName = QString::fromStdString(name);
        QVector<float> qFaceData;
        for (const float& value : faceData) {
            qFaceData.append(value);
        }
        stream << qName << "|| ";
        for (int i = 0; i < qFaceData.size(); ++i) {
            stream << qFaceData[i];
            if (i < qFaceData.size() - 1) {
                stream << ", ";
            }
        }
        stream << "\n";
        file.close();
    } else {
        std::cerr << "Error: Unable to open file for writing.\n";
    }
}

std::unordered_map<std::string, std::vector<float>> LoadDataFace::LoadUserData() {
    std::unordered_map<std::string, std::vector<float>> userData;
    QFile file("/home/duongnh/Projects/Qt/Face_Regconition/build/data.txt");
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        QString line;
        while (!in.atEnd()) {
            line = in.readLine();
            QStringList parts = line.split(" ");
            if (parts.size() == 2) {
                std::string name = parts[0].toStdString();
                std::vector<float> faceData;
                QStringList dataParts = parts[1].split(",");
                for (const auto &part : dataParts) {
                    faceData.push_back(part.toFloat());
                }
                userData[name] = faceData;
            }
        }
        file.close();
    } else {
        std::cerr << "Error: Unable to open file for reading.\n";
    }
    return userData;
}

float LoadDataFace::CalculateSimilarity(const std::vector<float> data1, const std::vector<float> data2) {
    float dot = 0.0, denomA = 0.0, denomB = 0.0;
    for (unsigned int i = 0; i < data1.size(); ++i) {
        dot += data1[i] * data2[i];
        denomA += data1[i] * data1[i];
        denomB += data2[i] * data2[i];
    }
    return dot / (sqrt(denomA) * sqrt(denomB));
}


