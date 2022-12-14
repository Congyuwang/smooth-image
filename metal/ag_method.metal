//
//  ag_method.metal
//  smooth-image
//
//  Created by Congyu WANG on 2022/12/13.
//

#include <metal_stdlib>
using namespace metal;

// Init:
// const alpha = 1.0 / (1.0 + 8.0 * mu)
// beta = 0.0
// t = 1.0
// x_old = copy(x)
//
// IMPORTANT: remember to reset r_new_norm_squared, r_norm_squared, dot to 0 on cpu before each run!!!
struct AgMethodBuffers {
    // private
    device half* x;
    device half* x_tmp;
    device half* x_old;
    device half* y;
    volatile device atomic_float* grad_norm;
    volatile device atomic_float* dot;
    volatile device atomic_float* diff_squared;
    device const half& alpha;
    device half& beta;
    device half& t;
    device const half* c;
    device const uint* row_offsets;
    device const uint* col_indices;
    device const half* values;
    device const half* original;
};

// reset dot, grad_norm
kernel void ag_step_0_reset_grad_norm(device AgMethodBuffers& buffers) {
    atomic_store_explicit(buffers.grad_norm, 0.0, memory_order_relaxed);
    atomic_store_explicit(buffers.dot, 0.0, memory_order_relaxed);
    atomic_store_explicit(buffers.diff_squared, 0.0, memory_order_relaxed);
}

// step_1
// copy x_tmp <- x

// step_2_1
// y^k+1
// x <- (1 + beta) * x - beta * x_old
kernel void ag_step_2_1_yk1(device AgMethodBuffers& buffers,
                            uint index [[thread_position_in_grid]]) {
    half one_plus_beta = 1.0 + buffers.beta;
    buffers.x[index] = one_plus_beta * buffers.x[index] - buffers.beta * buffers.x_old[index];
}

// step_2_2
// copy y <- x

// x <- B * x - c
kernel void ag_step_3_bx_minus_c(device AgMethodBuffers& buffers,
                                 uint index [[thread_position_in_grid]]) {
    const uint p1 = buffers.row_offsets[index + 1];
    half dot = 0.0;
    for (uint p = buffers.row_offsets[index]; p < p1; p++) {
        dot += buffers.values[p] * buffers.y[p];
    }
    buffers.x[index] = dot - buffers.c[index];
}

// ||DFx||
kernel void ag_step_4_grad_norm(device AgMethodBuffers& buffers,
                                uint index [[thread_position_in_grid]]) {
    half x = buffers.x[index];
    atomic_fetch_add_explicit(buffers.grad_norm, x * x, memory_order_relaxed);
}

// x = y - alpha * x
kernel void ag_step_5_update_x(device AgMethodBuffers& buffers,
                               uint index [[thread_position_in_grid]]) {
    buffers.x[index] = buffers.y[index] - buffers.alpha * buffers.x[index];
}

// update beta
kernel void ag_step_6_update_beta(device AgMethodBuffers& buffers) {
    half t = buffers.t;
    half t_new = 0.5 + 0.5 * sqrt(1.0 + 4.0 * t * t);
    buffers.beta = (t - 1.0) / t_new;
    buffers.t = t_new;
}

kernel void ag_step_7_diff_squared(device AgMethodBuffers& buffers,
                 uint index [[thread_position_in_grid]]) {
    half diff = buffers.x[index] - buffers.original[index];
    atomic_fetch_add_explicit(buffers.diff_squared, diff * diff, memory_order_relaxed);
}
