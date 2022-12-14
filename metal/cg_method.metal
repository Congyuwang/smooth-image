//
//  smooth_image.metal
//  smooth-image
//
//  Created by Congyu WANG on 2022/12/13.
//

#include <metal_stdlib>
using namespace metal;

// Init:
// r = c
// r = -r + b_mat * x
// p = -r
//
// IMPORTANT: remember to reset r_new_norm_squared, r_norm_squared, dot to 0 on cpu before each run!!!
struct CgMethodBuffers {
    // private
    volatile device atomic_float* r_new_norm_squared;
    volatile device atomic_float* r_norm_squared;
    volatile device atomic_float* dot;
    volatile device atomic_float* diff_squared;
    device half& alpha;
    device half& beta;
    device half* x;
    device half* bp;
    device half* p;
    device half* r;
    device const uint* row_offsets;
    device const uint* col_indices;
    device const half* values;
    device const half* original;
};

// execute in 1 * 1 * 1
kernel void cg_step_0_reset_alpha_beta(device CgMethodBuffers& buffers) {
    buffers.alpha = 0;
    buffers.beta = 0;
    atomic_store_explicit(buffers.r_new_norm_squared, 0.0, memory_order_relaxed);
    atomic_store_explicit(buffers.r_norm_squared, 0.0, memory_order_relaxed);
    atomic_store_explicit(buffers.dot, 0.0, memory_order_relaxed);
    atomic_store_explicit(buffers.diff_squared, 0.0, memory_order_relaxed);
}

// ||r||^2
kernel void cg_step_1_norm_squared2(device CgMethodBuffers& buffers,
                                    uint index [[thread_position_in_grid]]) {
    half r = buffers.r[index];
    atomic_fetch_add_explicit(buffers.r_norm_squared, r * r, memory_order_relaxed);
}

// bp = b_mat * p
kernel void cg_step_2_bp(device CgMethodBuffers& buffers,
                         uint index [[thread_position_in_grid]]) {
    const uint p1 = buffers.row_offsets[index + 1];
    half dot = 0.0;
    for (uint p = buffers.row_offsets[index]; p < p1; p++) {
        dot += buffers.values[p] * buffers.p[p];
    }
    buffers.bp[index] = dot;
}

// dot = p * bp
kernel void cg_step_3_1_dot_pbp(device CgMethodBuffers& buffers,
                                uint index [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(buffers.dot, buffers.bp[index] * buffers.p[index], memory_order_relaxed);
}

// alpha = 1 / dot
//
// execute in 1 * 1 * 1
kernel void cg_step_3_2_alpha(device CgMethodBuffers& buffers) {
    buffers.alpha = 1.0 / atomic_load_explicit(buffers.dot, memory_order_relaxed);
}

// x = x + alpha * p
//
// r = r + alpha * b * p
kernel void cg_step_4_update_x(device CgMethodBuffers& buffers,
                               uint index [[thread_position_in_grid]]) {
    half alpha = buffers.alpha;
    buffers.x[index] += alpha * buffers.p[index];
    buffers.r[index] += alpha * buffers.bp[index];
}

// ||r||^2
kernel void cg_step_5_1_new_norm_squared2(device CgMethodBuffers& buffers,
                                          uint index [[thread_position_in_grid]]) {
    half r = buffers.r[index];
    atomic_fetch_add_explicit(buffers.r_new_norm_squared, r * r, memory_order_relaxed);
}

// beta = r_new_norm_squared / r_norm_squared
//
// execute in 1 * 1 * 1
kernel void cg_step_5_2_beta(device CgMethodBuffers& buffers) {
    buffers.beta = atomic_load_explicit(buffers.r_new_norm_squared, memory_order_relaxed) / atomic_load_explicit(buffers.r_norm_squared, memory_order_relaxed);
}

// p = beta * p - r
kernel void cg_step_6_update_p(device CgMethodBuffers& buffers,
                               uint index [[thread_position_in_grid]]) {
    buffers.p[index] = buffers.beta * buffers.p[index] - buffers.r[index];
}

kernel void cg_step_7_diff_squared(device CgMethodBuffers& buffers,
                 uint index [[thread_position_in_grid]]) {
    half diff = buffers.x[index] - buffers.original[index];
    atomic_fetch_add_explicit(buffers.diff_squared, diff * diff, memory_order_relaxed);
}
