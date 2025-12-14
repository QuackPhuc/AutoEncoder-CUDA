#ifndef GPU_CORE_LAYOUT_CONVERT_H
#define GPU_CORE_LAYOUT_CONVERT_H

// NHWC to NCHW conversion
// Input: (batch, H, W, C) -> Output: (batch, C, H, W)
void launchNhwcToNchw(
    const float* d_nhwc, float* d_nchw,
    int batch, int height, int width, int channels);

// NCHW to NHWC conversion
// Input: (batch, C, H, W) -> Output: (batch, H, W, C)
void launchNchwToNhwc(
    const float* d_nchw, float* d_nhwc,
    int batch, int channels, int height, int width);

#endif // GPU_CORE_LAYOUT_CONVERT_H
