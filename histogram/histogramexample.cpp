#define STB_IMAGE_IMPLEMENTATION

#include <string>
#include "stb_image.h"
#include <array>
#include <stopwatch.h>
#include <iostream>

void cpuHistogram();


int main()
{
    cpuHistogram();

    return 0;
}

void cpuHistogram()
{
    int width;
    int height;
    int channels;
    auto imageData = stbi_load("resources/sample.png",&width,&height,&channels, STBI_rgb_alpha);
    constexpr size_t bufferSize{256*3};
    std::array<uint8_t,bufferSize> histogramBuffer{0};

    uint32_t offset = 0;

    Stopwatch stopwatch;
    stopwatch.Start();
    //histogram for R values
    for (int i = 0; i < width * height * 4; ++i)
    {
        int index = imageData[i];
        histogramBuffer[index + offset]++;
    }

    //histogram for G values
    offset += 256;
    for (int i = 1; i < width * height * 4; ++i)
    {
        int index = imageData[i];
        histogramBuffer[index + offset]++;
    }

    //histogram for B values
    offset += 256;
    for (int i = 2; i < width * height * 4; ++i)
    {
        int index = imageData[i];
        histogramBuffer[index + offset]++;
    }
    stopwatch.Stop();
    std::cout << "cpu histogram " << stopwatch.Time() << std::endl;

}