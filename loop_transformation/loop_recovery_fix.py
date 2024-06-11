import re
ParaVar = {
    "threadIdx.x": 1024,
    "blockIdx.x": 256,
    "coreId": 4;
    "clusterId": 4
}

cuda_paravar = ["threadIdx.x", "threadIdx.y", "blockIdx.x", "blockIdx.y"]
mlu_paravar = ["coreId", "clusterId"]


txz_loop = "for(int threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++){\n"
txy_loop = "for(int threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++){\n"
txx_loop = "for(int threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x++){\n"
bxx_loop = "for(int blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++){\n"
bxy_loop = "for(int blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++){\n"
bxz_loop = "for(int blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++){\n"


coreId_loop = "for(int coreId = 0; coreId < coreDim; coreId++) {\n"
clusterId_loop = "for(int clusterId = 0; clusterId < clusterDim; clusterId++) {\n"



def get_blockDim(cuda_code):
    """The re module in Python is used to write a regular expression 
    that matches the number inside the parentheses."""
    match = re.search(r'$(\d+)$', cuda_code)
    if match:
        thread_count = match.group(1)
        return thread_count
    else:
        return None

def loop_recovery():
    return


if __name__ == "__main__":
    




