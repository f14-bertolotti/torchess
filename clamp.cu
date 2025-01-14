#pragma once

__device__ __inline__ int clamp(long left, long right, long val) {
    // This function clamps the value between left and right, both included
    return max(left, min(right, val));
}
