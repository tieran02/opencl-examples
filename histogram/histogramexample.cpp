#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 300
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <array>
#include <stopwatch.h>
#include <iostream>
#include <util.h>

void cpuHistogram();
void computeHistogram();

const int num_pixels_per_work_item = 32;


int main()
{
    cpuHistogram();
    computeHistogram();
    return 0;
}

void cpuHistogram()
{
    int width;
    int height;
    int channels;
    auto imageData = stbi_load("resources/sample.png",&width,&height,&channels, STBI_rgb_alpha);
    constexpr size_t bufferSize{256*3};
    std::array<uint32_t,bufferSize> histogramBuffer{0};

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

void computeHistogram()
{
    int width;
    int height;
    auto imageData = stbi_load("resources/sample.png",&width,&height,nullptr, STBI_rgb_alpha);
    constexpr size_t bufferSize{257*3};
    std::array<uint32_t,bufferSize> histogramArray{0};
    uint32_t offset = 0;

    cl::NDRange global_work_size;
    cl::NDRange local_work_size;
    cl::NDRange partial_global_work_size;
    cl::NDRange partial_local_work_size;
    size_t workgroup_size;
    size_t num_groups;
    cl::NDRange gsize;

    cl_int err = CL_SUCCESS;

    try {

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cout << "Platform size 0\n";
            return;
        }

        // Print number of platforms and list of platforms
        std::cout << "Platform number is: " << platforms.size() << std::endl;
        std::string platformVendor;
        for (auto& platform : platforms)
        {
	        platform.getInfo(static_cast<cl_platform_info>(CL_PLATFORM_VENDOR), &platformVendor);
            std::cout << "Platform is by: " << platformVendor << std::endl;
        }

        cl_context_properties properties[] =
            {
                    CL_CONTEXT_PLATFORM,
                    reinterpret_cast<cl_context_properties>((platforms[0])()),
                    0
            };
        cl::Context context(CL_DEVICE_TYPE_ALL, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Print number of devices and list of devices
        std::cout << "Device number is: " << devices.size() << std::endl;
        for (unsigned int i = 0; i < devices.size(); ++i) {
            std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        }

        std::string kernalStr = LoadFromFile("resources/histogram.cl");
        const cl::Program::Sources source(1, std::make_pair(kernalStr.data(), kernalStr.size()));
        const cl::Program program = BuildProgram(context, devices, source);

        cl::Kernel histogram_rgba_unorm8(program, "histogram_partial_image_rgba_unorm8", &err);
        cl::Kernel histogram_sum_partial_results_unorm8(program, "histogram_sum_results_unorm8", &err);
    	
		//get max work-group size that can be used for histogram
        histogram_rgba_unorm8.getWorkGroupInfo(devices[0], CL_KERNEL_WORK_GROUP_SIZE, &workgroup_size);
        if(workgroup_size <= 256)
        {
            gsize = cl::NDRange(16, workgroup_size / 16);
        }
        else if (workgroup_size <= 1024)
        {
            gsize = cl::NDRange(workgroup_size / 16, 16);
        }
        else
        {
            gsize = cl::NDRange(workgroup_size / 32, 32);
        }

        local_work_size = cl::NDRange(gsize);

        global_work_size = cl::NDRange((width + gsize[0] - 1) / gsize[0],
									   (height + gsize[1] - 1) / gsize[1]);

        num_groups = global_work_size[0] * global_work_size[1];

        global_work_size = cl::NDRange(global_work_size[0] * gsize[0],
									   global_work_size[1] * gsize[1]);

        //load resources

		//copy image over to device (GPU)
        cl::Image2D input_image_unorm8 = cl::Image2D(context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
            width,
            height,
            0,
            static_cast<void*>(imageData));
    	
        //create partial image buffer
        cl::Buffer partial_histogram_buffer(context, CL_MEM_READ_WRITE, num_groups * 257 * 3 * sizeof(unsigned int), nullptr);

        histogram_rgba_unorm8.setArg(0, sizeof(cl::Memory), &input_image_unorm8);
        histogram_rgba_unorm8.setArg(1, sizeof(int), &num_pixels_per_work_item);
        histogram_rgba_unorm8.setArg(2, sizeof(cl::Memory), &partial_histogram_buffer);

        cl::Buffer histogramBuffer(context, CL_MEM_WRITE_ONLY, num_groups * 257 * 3 * sizeof(unsigned int), nullptr);
        histogram_sum_partial_results_unorm8.setArg(0, sizeof(cl::Memory), &partial_histogram_buffer);
        histogram_sum_partial_results_unorm8.setArg(1, sizeof(int), &num_groups);
        histogram_sum_partial_results_unorm8.setArg(2, sizeof(cl::Memory), &histogramBuffer);

        cl::CommandQueue queue(context, devices[0], 0, &err);
    	
        Stopwatch stopwatch;
        stopwatch.Start();

        queue.enqueueNDRangeKernel(histogram_rgba_unorm8,
            cl::NDRange(0),
            global_work_size,
            local_work_size);

        partial_global_work_size = cl::NDRange(256 * 3, (workgroup_size > 256) ? 256 : workgroup_size);
        cl::Event event;
        queue.enqueueNDRangeKernel(histogram_sum_partial_results_unorm8,
            cl::NDRange(0),
            partial_global_work_size,
            partial_local_work_size,
            nullptr,
            &event
        );
	
        queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, 257 * 3 * sizeof(unsigned int), histogramArray.data());

        stopwatch.Stop();
    	
        std::cout << "GPU histogram " << stopwatch.Time() << std::endl;
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
