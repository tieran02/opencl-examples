#define MAX_LOCAL_SIZE 256

__kernel void MergeSort(__global const int* unsorted, __global int* sorted, const int size)
{
    __local int local_buffer[MAX_LOCAL_SIZE * 2];

    int i = get_local_id(0); //index of workgroup
    int wg = get_local_size(0); //get workgroup size

    int offset = get_group_id(0) * wg;
    //move the in, out pointer to the block start
    unsorted += offset;
    sorted += offset;

    // Load block in local_buffer[WG]
    local_buffer[i] = unsorted[i];
    barrier(CLK_LOCAL_MEM_FENCE); // make sure local_buffer is entirely up to date

    for(int length=1; length < wg; length <<=1)
    {
        int iData = local_buffer[i];
        int iKey = iData;

        int indexInSeq = i & (length-1);  // index in our sequence in 0..length-1
        int sibling = (i - indexInSeq) ^ length; // beginning of the sibling sequence
        int pos = 0;

        for (int inc=length;inc>0;inc>>=1) // increment for dichotomic search
        {
            int j = sibling+pos+inc-1;
            int jKey = local_buffer[j];
            bool smaller = (jKey < iKey) || ( jKey == iKey && j < i );
            pos += (smaller)?inc:0;
            pos = min(pos,length);
        }
        int bits = 2*length-1; // mask for destination
        int dest = ((indexInSeq + pos) & bits) | (i & ~bits); // destination index in merged sequence
        barrier(CLK_LOCAL_MEM_FENCE);
        local_buffer[dest] = iData;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write output
    sorted[i] = local_buffer[i];
}