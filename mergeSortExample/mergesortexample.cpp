#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 300
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <array>
#include <vector>
#include <stopwatch.h>
#include <iostream>
#include <util.h>
#include <algorithm>

template <size_t size> std::vector<int> randomNumbers();
void NormalMergeSort();
void cpuMergeSort(int* array, int leftIndex, int rightIndex);
void cpuMerge(int* array, int leftIndex,int midIndex, int rightIndex);
void stdSort();

void ComputeMergeSort();

constexpr int array_size = 1024000;

int main()
{
    stdSort();
    ComputeMergeSort();
    NormalMergeSort();
    return 0;
}

#pragma region cpuMergeSort
void NormalMergeSort()
{
    std::vector<int> array = randomNumbers<array_size>();

    Stopwatch stopwatch;
    stopwatch.Start();
    cpuMergeSort(array.data(),0,array.size()-1);

    stopwatch.Stop();
    std::cout << array_size << " int int cpu merge sort " << stopwatch.Time() << std::endl;
}

void cpuMergeSort(int *array, int leftIndex, int rightIndex)
{
    if(leftIndex>=rightIndex)
    {
        return;//returns recursively
    }
    int midIndex = (leftIndex+rightIndex-1) / 2;
    cpuMergeSort(array,leftIndex,midIndex);
    cpuMergeSort(array,midIndex+1,rightIndex);
    cpuMerge(array,leftIndex,midIndex,rightIndex);
}

void cpuMerge(int *array, int leftIndex, int midIndex, int rightIndex)
{
    const int leftArraySize = midIndex - leftIndex + 1;
    const int rightArraySize = rightIndex - midIndex;

    //create temp vectors
    std::vector<int> L(leftArraySize);
    std::vector<int> R(rightArraySize);

    //copy data to temp arrays
    memcpy(&L[0], &array[leftIndex], leftArraySize * sizeof(int));
    memcpy(&R[0], &array[midIndex+1], rightArraySize * sizeof(int));

    //merge temp arrays back into array[l..r]
    int i =0, j = 0, k = leftIndex;

    while (i < leftArraySize && j < rightArraySize)
    {
        if (L[i] <= R[j])
        {
            array[k] = L[i];
            i++;
        }
        else {
            array[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of
    // L[], if there are any
    while (i < leftArraySize)
    {
        array[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of
    // R[], if there are any
    while (j < rightArraySize)
    {
        array[k] = R[j];
        j++;
        k++;
    }
}
#pragma endregion

#pragma region gpuMergeSort

void ComputeMergeSort()
{
    constexpr int local_work_size = 16;
    std::vector<int> array = randomNumbers<array_size>();
    std::vector<int> sortedArray(array_size);

    cl_int err = CL_SUCCESS;
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            //std::cout << "Platform size 0\n";
            return;
        }

        // Print number of platforms and list of platforms
        //std::cout << "Platform number is: " << platforms.size() << std::endl;
        std::string platformVendor;
        for (unsigned int i = 0; i < platforms.size(); ++i) {
            platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
            //std::cout << "Platform is by: " << platformVendor << std::endl;
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
        //std::cout << "Device number is: " << devices.size() << std::endl;
        for (unsigned int i = 0; i < devices.size(); ++i) {
            //std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        }

        //create buffers
        cl::Buffer bufferA(context, CL_MEM_READ_WRITE, array.size() * sizeof(int), nullptr, &err);
        cl::Buffer bufferB(context, CL_MEM_READ_WRITE, array.size() * sizeof(int), nullptr, &err);

        //copy unsorted data to buffer
        cl::CommandQueue queue(context, devices[0], 0, &err);
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, array.size() * sizeof(int), array.data());

        std::string kernalStr = LoadFromFile("kernels/mergeSort.cl");
        const cl::Program::Sources source(1, std::make_pair(kernalStr.data(), kernalStr.size()));
        const cl::Program program = BuildProgram(context, devices, source);

        cl::Kernel sortKernel(program, "MergeSort", &err);
        sortKernel.setArg(0, sizeof(bufferA), &bufferA);
        sortKernel.setArg(1, sizeof(bufferB), &bufferB);
        sortKernel.setArg(2, sizeof(int), &array_size);

        cl::Kernel mergeKernel(program, "Merge", &err);
        mergeKernel.setArg(3,sizeof(int), &array_size);

        //get max work-group size that can be used for sortKernel
        size_t workgroup_size;
        sortKernel.getWorkGroupInfo(devices[0], CL_KERNEL_WORK_GROUP_SIZE, &workgroup_size);


        Stopwatch stopwatch;
        stopwatch.Start();

        cl::NDRange globalWorkSize(array_size);
        cl::NDRange localWorkSize(local_work_size);

        queue.enqueueNDRangeKernel(
                sortKernel,
                cl::NDRange(0),
                globalWorkSize,
                localWorkSize);

        //swap buffers, so partial sorted becomes the primary buffer
        std::swap(bufferA,bufferB);


        //now continue with global merge
        unsigned int stride = local_work_size;

        for(; stride <= array_size; stride <<= 1)
        {
            //get work sizes
            int neededWorkers = array_size / stride;

            localWorkSize = cl::NDRange(std::min(local_work_size, neededWorkers));
            globalWorkSize = GetGlobalSize(neededWorkers, localWorkSize[0]);

            mergeKernel.setArg(0, sizeof(cl::Buffer), &bufferA);
            mergeKernel.setArg(1, sizeof(cl::Buffer), &bufferB);
            mergeKernel.setArg(2, sizeof(cl_uint), &stride);

            queue.enqueueNDRangeKernel(
                    mergeKernel,
                    cl::NDRange(0),
                    globalWorkSize,
                    localWorkSize);

            //swap buffers, so partial sorted becomes the primary buffer
            std::swap(bufferA,bufferB);
        }

        cl::Event event;
        queue.enqueueReadBuffer(bufferA, true, 0, array_size * sizeof(int), sortedArray.data(), nullptr, &event);
        //wait for event to finish
        event.wait();
        queue.finish();

        stopwatch.Stop();
        std::cout << array_size << " int compute merge sort " << stopwatch.Time() << std::endl;
    }
    catch (cl::Error err)
    {
        std::cerr
                << "ERROR: "
                << err.what()
                << "("
                << getErrorString(err.err())
                << ")"
                << std::endl;
    }
}


#pragma endregion

void stdSort()
{
    std::vector<int> array = randomNumbers<array_size>();

    Stopwatch stopwatch;
    stopwatch.Start();
    std::sort(array.begin(), array.end());

    stopwatch.Stop();
    std::cout << array_size << " int int std sort " << stopwatch.Time() << std::endl;
}

template<size_t size>
std::vector<int> randomNumbers()
{
    std::vector<int> array(size);

    for (int i = 0; i < size; ++i)
    {
        array[i] = rand() % std::numeric_limits<int>::max();
    }

    return array;
}

