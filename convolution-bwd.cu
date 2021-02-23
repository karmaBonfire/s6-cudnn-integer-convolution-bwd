#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cudnn.h>

template <typename T, typename BufferT>
__global__ void convolutionBackwardFilterExNHWC(T* x, T* dy, T* dw, int channels, int dyH, int dyW, int xH, int xW, float alpha, float beta) {
    // LAYOUT: NHWC - height first

    int dwH = blockDim.x;
    int dwW = blockDim.y;
    int C = channels;

    int n = blockIdx.x;
    int dwh = threadIdx.x;
    int dww = threadIdx.y;

    BufferT buffer = 0;

    size_t dyIdx, xIdx, dwBaseIdx;

    // NHWC-tensor indices formula:
    // ---------------------------
    // Idx = c + C*w + C*W*h + C*W*H*n,
    // ---------------------------
    // where c, w, h, n -- current dimension indices,
    // C, W, H, N -- dimension sizes

    dwBaseIdx = 0 
        + C * dww 
        + C * dwW * dwh 
        + C * dwW * dwH * n; // full index depends on a channel index

    for (int c = 0; c < channels; c++) {
        buffer = 0;

        for (int row = 0; row < dyH; row++) {
            for (int col = 0; col < dyW; col++) {
                dyIdx = c 
                    + C * col 
                    + C * dyW * row 
                    + C * dyW * dyH * n;
                xIdx = c 
                    + C * (col + dww) 
                    + C * xW * (row + dwh) 
                    + C * xW * xH * n;
                buffer += dy[dyIdx] * x[xIdx];
            }
        }

        // scaling, see https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters ("Type Conversion" section)
        dw[dwBaseIdx + c] = (T)(beta * ((float)dw[dwBaseIdx + c]) + (float)buffer);
    }
}

cudnnStatus_t cudnnConvolutionBackwardFilterEx(
    cudnnHandle_t                       handle,
    const void*                         alpha,
    const cudnnTensorDescriptor_t       xDesc,
    const void*                         x,
    const cudnnTensorDescriptor_t       dyDesc,
    const void*                         dy,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionBwdFilterAlgo_t     algo,
    void*                               workSpace,
    size_t                              workSpaceSizeInBytes,
    const void*                         beta,
    const cudnnFilterDescriptor_t       dwDesc,
    void*                               dw
) {
    if (!handle || !xDesc || !dyDesc || !convDesc || !dwDesc || !x || !dy || !dw || !alpha || !beta)
        return CUDNN_STATUS_BAD_PARAM;

    cudnnStatus_t subroutineStatus;
    cudnnDataType_t xType, dyType, dwType;
    cudnnTensorFormat_t dwFormat;
    int strides[8];
    int xn, xc, xh, xw;
    int dyn, dyc, dyh, dyw;
    int dwk, dwc, dwh, dww;

    // Get tensor info

    subroutineStatus = cudnnGetTensor4dDescriptor(
        xDesc, &xType, &xn, &xc, &xh, &xw,
        strides + 0, strides + 1, strides + 2, strides + 3
    );

    if (subroutineStatus != CUDNN_STATUS_SUCCESS)
        return subroutineStatus;

    subroutineStatus = cudnnGetTensor4dDescriptor(
        dyDesc, &dyType, &dyn, &dyc, &dyh, &dyw,
        strides + 4, strides + 5, strides + 6, strides + 7
    );

    if (subroutineStatus != CUDNN_STATUS_SUCCESS)
        return subroutineStatus;

    subroutineStatus = cudnnGetFilter4dDescriptor(
        dwDesc, &dwType, &dwFormat,
        &dwk, &dwc, &dwh, &dww
    );

    if (subroutineStatus != CUDNN_STATUS_SUCCESS)
        return subroutineStatus;

    bool int8_compatible = (xType == CUDNN_DATA_INT8) && (dyType == CUDNN_DATA_INT8) && (dwType == CUDNN_DATA_INT8);
    bool uint8_compatible = (xType == CUDNN_DATA_UINT8) && (dyType == CUDNN_DATA_UINT8) && (dwType == CUDNN_DATA_UINT8);
    bool int32_compatible = (xType == CUDNN_DATA_INT32) && (dyType == CUDNN_DATA_INT32) && (dwType == CUDNN_DATA_INT32);

    if (int8_compatible || uint8_compatible || int32_compatible) {
        int pad_h, pad_w, u, v, dilation_h, dilation_w;
        cudnnConvolutionMode_t conv_mode;
        cudnnDataType_t compute_type;

        // Get convolution info

        subroutineStatus = cudnnGetConvolution2dDescriptor(
            convDesc, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w,
            &conv_mode, &compute_type
        );

        if (subroutineStatus != CUDNN_STATUS_SUCCESS)
            return subroutineStatus;

        //
        // Sanity checks
        //

        // Not implemented
        if (conv_mode == CUDNN_CROSS_CORRELATION) return CUDNN_STATUS_NOT_SUPPORTED;

        // Only _ALGO_1 (matrix product, non-atomic) is supported
        if (algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) return CUDNN_STATUS_NOT_SUPPORTED;

        // Wrong, see https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_765/cudnn-api/index.html#cudnnConvolutionForward 
        //// Tensor types and compute types must match
        //if (uint8_compatible && (compute_type != CUDNN_DATA_UINT8)) return CUDNN_STATUS_BAD_PARAM;
        //if (int8_compatible && (compute_type != CUDNN_DATA_INT8)) return CUDNN_STATUS_BAD_PARAM;
        //if (int32_compatible && (compute_type != CUDNN_DATA_INT32)) return CUDNN_STATUS_BAD_PARAM;

        // Instead, check for INT32 compute type
        if (compute_type != CUDNN_DATA_INT32) return CUDNN_STATUS_NOT_SUPPORTED;

        // Initialize device stream
        cudaStream_t stream = NULL;
        subroutineStatus = cudnnGetStream(handle, &stream);
        if (subroutineStatus != CUDNN_STATUS_SUCCESS)
            return subroutineStatus;

        // Run on k blocks, each with h*w threads, having 0 shared memory per block, on initialized device stream
        dim3 blocks = dim3(dwk);
        dim3 threads = dim3(dwh, dww);
        int shared_bytes = 0;

        if (uint8_compatible) {
            convolutionBackwardFilterExNHWC<uint8_t, int32_t> <<< blocks, threads, shared_bytes, stream >>> (
                (uint8_t*)x,
                (uint8_t*)dy,
                (uint8_t*)dw,
                xc,
                dyh, dyw,
                xh, xw,
                *((float*)alpha), *((float*)beta)
                );
        }
        else if (int8_compatible) {
            convolutionBackwardFilterExNHWC<int8_t, int32_t> <<< blocks, threads, shared_bytes, stream >>> (
                (int8_t*)x,
                (int8_t*)dy,
                (int8_t*)dw,
                xc,
                dyh, dyw,
                xh, xw,
                *((float*)alpha), *((float*)beta)
                );
        }
        else if (int8_compatible) {
            convolutionBackwardFilterExNHWC<int32_t, int32_t> <<< blocks, threads, shared_bytes, stream >>> (
                (int32_t*)x,
                (int32_t*)dy,
                (int32_t*)dw,
                xc,
                dyh, dyw,
                xh, xw,
                *((float*)alpha), *((float*)beta)
                );
        }

		cudaError_t cudaerr = cudaDeviceSynchronize();
    return (cudaerr == cudaSuccess) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
    }
    else {
        return cudnnConvolutionBackwardFilter(
            handle,
            alpha, xDesc, x,
            dyDesc, dy,
            convDesc, algo,
            workSpace, workSpaceSizeInBytes,
            beta, dwDesc, dw
        );
    }
}

