#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 300
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <string>
#include <util.h>
#include "stopwatch.h"

constexpr size_t ARRAY_SIZE = 65536;
constexpr size_t localItemSize = 512;

int computeAdd()
{
    std::array<int, ARRAY_SIZE> A;
    std::array<int, ARRAY_SIZE> B;
    std::array<int, ARRAY_SIZE> C;
    Stopwatch stopwatch;

    /* initialize random seed: */
    srand(time(nullptr));

    //fill vector with random numbers
    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        A[i] = rand() % 100 + 1;
        B[i] = rand() % 100 + 1;
    }


    cl_int err = CL_SUCCESS;
    try {

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return -1;
        }

        // Print number of platforms and list of platforms
        std::cout << "Platform number is: " << platforms.size() << std::endl;
        std::string platformVendor;
        for (unsigned int i = 0; i < platforms.size(); ++i) {
            platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
            std::cout << "Platform is by: " << platformVendor << std::endl;
        }

        cl_context_properties properties[] =
        {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)(platforms[0])(),
                0
        };
        cl::Context context(CL_DEVICE_TYPE_ALL, properties);


        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Print number of devices and list of devices
        std::cout << "Device number is: " << devices.size() << std::endl;
        for (unsigned int i = 0; i < devices.size(); ++i) {
            std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        }

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY, ARRAY_SIZE * sizeof(int), nullptr, &err);
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY, ARRAY_SIZE * sizeof(int), nullptr, &err);
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, ARRAY_SIZE * sizeof(int), nullptr, &err);

        cl::CommandQueue queue(context, devices[0], 0, &err);
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, A.size() * sizeof(int), A.data());
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, B.size() * sizeof(int), B.data());

        std::string kernalStr = LoadFromFile("kernals/VectorAdd.cl");
        cl::Program::Sources source(1, std::make_pair(kernalStr.data(), kernalStr.size()));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);

        cl::Kernel kernel(program_, "vector_add", &err);
        kernel.setArg(0, sizeof(bufferA), &bufferA);
        kernel.setArg(1, sizeof(bufferB), &bufferB);
        kernel.setArg(2, sizeof(bufferC), &bufferC);

        stopwatch.Start();
    	
        queue.enqueueNDRangeKernel(
            kernel,
            cl::NDRange(0),
            cl::NDRange(ARRAY_SIZE),
            cl::NDRange(localItemSize));


        cl::Event event;
        queue.enqueueReadBuffer(bufferC, true, 0, ARRAY_SIZE * sizeof(int), C.data(), nullptr, &event);
        //wait for event to finish
        event.wait();

        stopwatch.Stop();
        std::cout << "compute add " << stopwatch.Time() << std::endl;

    }
    catch (cl::Error err) {
        std::cerr
            << "ERROR: "
            << err.what()
            << "("
            << err.err()
            << ")"
            << std::endl;
    }
    return EXIT_SUCCESS;
}

void add()
{
    std::array<int, ARRAY_SIZE> A;
    std::array<int, ARRAY_SIZE> B;
    std::array<int, ARRAY_SIZE> C;
    Stopwatch stopwatch;

    /* initialize random seed: */
    srand(time(nullptr));

    //fill vector with random numbers
    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        A[i] = rand() % 100 + 1;
        B[i] = rand() % 100 + 1;
    }

    stopwatch.Start();
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        C[i] = A[i] + B[i];
    }

    stopwatch.Stop();
    std::cout << "add " << stopwatch.Time() << std::endl;
}

int main(void)
{
    add();
    computeAdd();
}
