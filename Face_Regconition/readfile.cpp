#include "readfile.h"
#include "utils.h"
ReadFile::ReadFile() {}

void ReadFile::ReadData(std::string name, std::vector<float> vecdb){
    QFile file("/home/duongnh/Projects/Qt/Face_Regconition/build/data.txt");
    // std::string name;
    LOG_C("Input your name: " << name);
    std::cin >> name;
    LOG_C("Name input to file: " << name);
    QString qName = QString::fromStdString(name);
    QVector<float> qVec;
    for (const float& value : vecdb) {
        qVec.append(value);
    }
    bool nameExists = false;
    
    if(file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append))
    {
        QTextStream in(&file);
        QString qLine;
        while (!in.atEnd()) {
            qLine = in.readLine();
            if (qLine == qName){
                nameExists = true;
                LOG_C("Error: Name already exists in the file.");
                break;
            }
        }
        file.close();
    }
    if(nameExists) {
        LOG_C("Error: Name already exists in the file.");
    } else{
        if(file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append))
        {
            QTextStream stream(&file);
            stream << qName << "||" ;
            for (int i = 0; i < qVec.size(); ++i) {
                stream << qVec[i];
                if (i < qVec.size() - 1) {
                    stream << ",";
                }
            }
            stream << "\n";
            file.close();
            LOG_C("Write done!");
        }
    }
}
