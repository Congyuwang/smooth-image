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
    // cpu cache
    volatile device atomic_float* r_new_norm_squared [[id(0)]];
    volatile device atomic_float* r_norm_squared [[id(1)]];
    volatile device atomic_float* dot [[id(2)]];
    volatile device atomic_float* diff_squared [[id(3)]];
    device half* alpha [[id(4)]];
    device half* beta [[id(5)]];

    // private
    device half* x [[id(6)]];
    device half* bp [[id(7)]];
    device half* p [[id(8)]];
    device half* r [[id(9)]];
    device const uint* row_offsets [[id(10)]];
    device const uint* col_indices [[id(11)]];
    device const half* values [[id(12)]];
    device const half* original [[id(13)]];
};

kernel void cg_init(device CgMethodBuffers& buffers,
                    uint index [[thread_position_in_grid]]) {
    const uint p1 = buffers.row_offsets[index + 1];
    half dot = 0.0;
    for (uint p = buffers.row_offsets[index]; p < p1; p++) {
        dot += buffers.values[p] * buffers.x[buffers.col_indices[p]];
    }
    buffers.r[index] = -buffers.r[index] + dot;
    buffers.p[index] = -buffers.r[index];
}

// execute in 1 * 1 * 1
kernel void cg_step_0_reset_alpha_beta(device CgMethodBuffers& buffers) {
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
        dot += buffers.values[p] * buffers.p[buffers.col_indices[p]];
    }
    buffers.bp[index] = dot;
}

// dot = p * bp
kernel void cg_step_3_1_dot_pbp(device CgMethodBuffers& buffers,
                                uint index [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(buffers.dot, buffers.bp[index] * buffers.p[index], memory_order_relaxed);
}

// alpha = ||r||2 / (p' * b * p)
//
// execute in 1 * 1 * 1
kernel void cg_step_3_2_alpha(device CgMethodBuffers& buffers) {
    *buffers.alpha = atomic_load_explicit(buffers.r_norm_squared, memory_order_relaxed) /
    atomic_load_explicit(buffers.dot, memory_order_relaxed);
}

// x = x + alpha * p
//
// r = r + alpha * b * p
kernel void cg_step_4_update_x(device CgMethodBuffers& buffers,
                               uint index [[thread_position_in_grid]]) {
    half alpha = *buffers.alpha;
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
    *buffers.beta = atomic_load_explicit(buffers.r_new_norm_squared, memory_order_relaxed) / atomic_load_explicit(buffers.r_norm_squared, memory_order_relaxed);
}

// p = beta * p - r
kernel void cg_step_6_update_p(device CgMethodBuffers& buffers,
                               uint index [[thread_position_in_grid]]) {
    buffers.p[index] = *buffers.beta * buffers.p[index] - buffers.r[index];
}

kernel void cg_step_7_diff_squared(device CgMethodBuffers& buffers,
                                   uint index [[thread_position_in_grid]]) {
    half diff = buffers.x[index] - buffers.original[index];
    atomic_fetch_add_explicit(buffers.diff_squared, diff * diff, memory_order_relaxed);
}
