#ifndef READFILE_H
#define READFILE_H
#include <QTextStream>
#include <QFile>
#include<QVector>

class ReadFile
{
public:
    ReadFile();
    void ReadData(std::string name, std::vector<float> vecdb);
};

#endif // READFILE_H
