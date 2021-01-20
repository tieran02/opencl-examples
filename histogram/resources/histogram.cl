__kernel void hello(void)
{

}

__kernel void histogram_partial_image_rgba_unorm8(image2d_t img, int num_pixels_per_work_item, global uint *histogram)
{
    int local_size = get_local_size(0) * get_local_size(1);
    int image_width = get_image_width(img);
    int image_height = get_image_height(img);
    int group_index = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 * 3;

    int x = get_global_id(0);
    int y = get_global_id(1);

    local uint tmp_histogram[256 * 3];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);

    int j = 256 * 3;
    int index = 0;

    //clear local buffer for partial histogram
    do 
    {
        if(tid < j)
            tmp_histogram[index+tid] = 0;

        j -= local_size;
        index += local_size;
    } while (j > 0);

    //The barrier function will either flush any variables stored in local memory or queue a memory fence to ensure correct ordering of memory operations to local memory.
    barrier(CLK_LOCAL_MEM_FENCE);

    int i, idx;
    for(i=0, idx=x; i < num_pixels_per_work_item; i++, idx += get_global_size(0))
    {
            
        //create partial histogram
        if((idx < image_width) && (y < image_height))
        {
            sampler_t _sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE |  CLK_FILTER_NEAREST;
            float4 clr = read_imagef(img, 
                                    _sampler,
                                    (float2)(idx,y));

            uchar index_x = convert_uchar_sat(clr.x * 255.0f);
            uchar index_y = convert_uchar_sat(clr.y * 255.0f);
            uchar index_z = convert_uchar_sat(clr.z * 255.0f);

            //atomic increment
            atomic_inc(&tmp_histogram[index_x]);
            atomic_inc(&tmp_histogram[index_y + 256]);
            atomic_inc(&tmp_histogram[index_z + 512]);
        }

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //copy partial histogram to correct location in histogram given by the group index
    if(local_size >= (256 * 3))
    {
        if(tid < (256 * 3))
            histogram[group_index + tid] = tmp_histogram[tid];
    }
    else
    {
        int j = 256 * 3;
        int index = 0;
        do 
        {
            if(tid < j)
                histogram[group_index + index + tid] = tmp_histogram[index + tid];

            j -= local_size;
            index += local_size;
        } while (j > 0);
    }
}

__kernel void histogram_sum_results_unorm8(global uint *partial_histogram,
                                           int num_groups,
                                           global uint *histogram)
{
    int tid = (int)get_local_id(0);
    int group_index;
    int n = num_groups;
    local uint tmp_histogram[256*3];

    tmp_histogram[tid] = partial_histogram[tid];

    group_index = 256*3;
    while(--n > 0)
    {
        tmp_histogram[tid] += partial_histogram[group_index + tid];
        group_index += 256 * 3;
    }
    histogram[tid] = tmp_histogram[tid];
}