use half::f16;
use metal::*;
use std::cmp::min;
use std::ffi::c_void;
use std::mem;
use std::mem::{size_of, size_of_val, transmute};

const LIB: &[u8] = include_bytes!("./metallib/gpu.metallib");

pub struct GpuLib {
    device: Device,
    default_queue: CommandQueue,
    cg_step_0_reset_alpha_beta: ComputePipelineState,
    cg_step_1_norm_squared2: ComputePipelineState,
    cg_step_2_bp: ComputePipelineState,
    cg_step_3_1_dot_pbp: ComputePipelineState,
    cg_step_3_2_alpha: ComputePipelineState,
    cg_step_4_update_x: ComputePipelineState,
    cg_step_5_1_new_norm_squared2: ComputePipelineState,
    cg_step_5_2_beta: ComputePipelineState,
    cg_step_6_update_p: ComputePipelineState,
    ag_step_0_reset_grad_norm: ComputePipelineState,
    ag_step_2_1_yk1: ComputePipelineState,
    ag_step_3_bx_minus_c: ComputePipelineState,
    ag_step_4_grad_norm: ComputePipelineState,
    ag_step_5_update_x: ComputePipelineState,
    ag_step_6_update_beta: ComputePipelineState,
}

fn init_module() -> GpuLib {
    let mut device = Device::system_default().expect("failed to find gpu");
    if !device.has_unified_memory() {
        panic!("requires unified memory!");
    }
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_data(LIB)
        .expect("invalid metal library");

    GpuLib {
        device,
        default_queue: queue,
        cg_step_0_reset_alpha_beta: get_func("cg_step_0_reset_alpha_beta", &lib, &device),
        cg_step_1_norm_squared2: get_func("cg_step_1_norm_squared2", &lib, &device),
        cg_step_2_bp: get_func("cg_step_2_bp", &lib, &device),
        cg_step_3_1_dot_pbp: get_func("cg_step_3_1_dot_pbp", &lib, &device),
        cg_step_3_2_alpha: get_func("cg_step_3_2_alpha", &lib, &device),
        cg_step_4_update_x: get_func("cg_step_4_update_x", &lib, &device),
        cg_step_5_1_new_norm_squared2: get_func("cg_step_5_1_new_norm_squared2", &lib, &device),
        cg_step_5_2_beta: get_func("cg_step_5_2_beta", &lib, &device),
        cg_step_6_update_p: get_func("cg_step_6_update_p", &lib, &device),
        ag_step_0_reset_grad_norm: get_func("ag_step_0_reset_grad_norm", &lib, &device),
        ag_step_2_1_yk1: get_func("ag_step_2_1_yk1", &lib, &device),
        ag_step_3_bx_minus_c: get_func("ag_step_3_bx_minus_c", &lib, &device),
        ag_step_4_grad_norm: get_func("ag_step_4_grad_norm", &lib, &device),
        ag_step_5_update_x: get_func("ag_step_5_update_x", &lib, &device),
        ag_step_6_update_beta: get_func("ag_step_6_update_beta", &lib, &device),
    }
}

fn get_func(name: &str, lib: &Library, device: &Device) -> ComputePipelineState {
    let func = lib
        .get_function(name, None)
        .expect(&format!("failed to load function ({name})"));
    device
        .new_compute_pipeline_state_with_function(&func)
        .expect(&format!("failed to get pipeline of ({name})"))
}

impl GpuLib {
    pub fn init() -> Self {
        init_module()
    }

    pub fn empty_private_buffer<T: Sized>(&self, size: usize) -> Buffer {
        let buf_size = (size_of::<T>() * size) as u64;
        self.device
            .new_buffer(buf_size, MTLResourceOptions::StorageModePrivate)
    }

    pub fn private_buffer_u32(&self, data: &[usize]) -> (Buffer, &CommandBufferRef) {
        let buf_size = (size_of::<u32>() * data.len()) as u64;
        let buf = data
            .into_iter()
            .map(|f| *f as u32)
            .collect::<Vec<_>>();
        let buf = self.device.new_buffer_with_data(
            buf.as_ptr() as *const c_void,
            buf_size,
            MTLResourceOptions::StorageModeShared,
        );
        let private_buf = self
            .device
            .new_buffer(buf_size, MTLResourceOptions::StorageModePrivate);
        let commands = self.default_queue.new_command_buffer();
        self.copy_from_buffer(&buf, &private_buf, commands);
        commands.commit();
        (private_buf, commands)
    }

    /// returns both the buffer and the copy command, call `wait`.
    pub fn private_buffer_f16(&self, data: &[f32]) -> (Buffer, &CommandBufferRef) {
        let buf_size = (size_of::<f16>() * data.len()) as u64;
        let buf = data
            .into_iter()
            .map(|f| f16::from_f32(*f))
            .collect::<Vec<_>>();
        let buf = self.device.new_buffer_with_data(
            buf.as_ptr() as *const c_void,
            buf_size,
            MTLResourceOptions::StorageModeShared,
        );
        let private_buf = self
            .device
            .new_buffer(buf_size, MTLResourceOptions::StorageModePrivate);
        let commands = self.default_queue.new_command_buffer();
        self.copy_from_buffer(&buf, &private_buf, commands);
        commands.commit();
        (private_buf, commands)
    }

    pub fn shared_buffer(&self, data: &[f32]) -> Buffer {
        let buf_size = (size_of::<f16>() * data.len()) as u64;
        let buf = data
            .into_iter()
            .map(|f| f16::from_f32(*f))
            .collect::<Vec<_>>();
        let buf = self.device.new_buffer_with_data(
            buf.as_ptr() as *const c_void,
            buf_size,
            MTLResourceOptions::StorageModeShared,
        );
        buf
    }

    // for intel chip mac
    pub fn managed_buffer(&self, data: &[f32]) -> Buffer {
        let buf_size = (size_of::<f16>() * data.len()) as u64;
        let buf = data
            .into_iter()
            .map(|f| f16::from_f32(*f))
            .collect::<Vec<_>>();
        let buf = self.device.new_buffer_with_data(
            buf.as_ptr() as *const c_void,
            buf_size,
            MTLResourceOptions::StorageModeManaged,
        );
        buf
    }

    pub fn copy_from_buffer(&self, src: &Buffer, dest: &Buffer, commands: &CommandBufferRef) {
        let blit_pass = commands.new_blit_command_encoder();
        blit_pass.copy_from_buffer(src, 0, dest, 0, buf_size);
        blit_pass.end_encoding();
    }

    pub fn init_ag_argument() {

    }

    pub fn axpy(&self, a: f16, x: &Buffer, y: &Buffer, commands: &CommandBufferRef) {
        let compute = commands.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&self.axpy_fun);
        let a = a.to_le_bytes();
        compute.set_bytes(0, size_of_val(&a) as u64, a.as_ptr() as *const c_void);
        compute.set_buffer(1, Some(x), 0);
        compute.set_buffer(2, Some(y), 0);
        let threads_per_thread_group = min(
            x.length(),
            self.axpy_fun.max_total_threads_per_threadgroup(),
        );
        compute.dispatch_threads(
            MTLSize::new(x.length(), 0, 0),
            MTLSize::new(threads_per_thread_group, 0, 0),
        );
        compute.end_encoding();
    }

    pub fn axpby(&self, a: f16, x: &Buffer, b: f16, y: &Buffer, commands: &CommandBufferRef) {
        let compute = commands.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&self.axpby_fun);
        let a = a.to_le_bytes();
        let b = b.to_le_bytes();
        compute.set_bytes(0, size_of_val(&a) as u64, a.as_ptr() as *const c_void);
        compute.set_buffer(1, Some(x), 0);
        compute.set_bytes(2, size_of_val(&b) as u64, b.as_ptr() as *const c_void);
        compute.set_buffer(3, Some(y), 0);
        let threads_per_thread_group = min(
            x.length(),
            self.axpby_fun.max_total_threads_per_threadgroup(),
        );
        compute.dispatch_threads(
            MTLSize::new(x.length(), 0, 0),
            MTLSize::new(threads_per_thread_group, 0, 0),
        );
        compute.end_encoding();
    }

    pub fn dot(&self, x: &Buffer, y: &Buffer, output: &Buffer, commands: &CommandBufferRef) {
        let compute = commands.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&self.dot_fun);
        compute.set_buffer(0, Some(x), 0);
        compute.set_buffer(1, Some(y), 0);
        compute.set_buffer(2, Some(output), 0);
        let threads_per_thread_group =
            min(x.length(), self.dot_fun.max_total_threads_per_threadgroup());
        compute.dispatch_threads(
            MTLSize::new(x.length(), 1, 1),
            MTLSize::new(threads_per_thread_group, 1, 1),
        );
        compute.end_encoding();
    }

    pub fn norm_squared(&self, x: &Buffer, output: &Buffer, commands: &CommandBufferRef) {
        let compute = commands.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&self.norm_squared_fun);
        compute.set_buffer(0, Some(x), 0);
        compute.set_buffer(1, Some(output), 0);
        let threads_per_thread_group = min(
            x.length(),
            self.norm_squared_fun.max_total_threads_per_threadgroup(),
        );
        compute.dispatch_threads(
            MTLSize::new(x.length(), 1, 1),
            MTLSize::new(threads_per_thread_group, 1, 1),
        );
        compute.end_encoding();
    }

    pub fn dist_squared(
        &self,
        x: &Buffer,
        y: &Buffer,
        output: &Buffer,
        commands: &CommandBufferRef,
    ) {
        let compute = commands.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&self.dist_squared_fun);
        compute.set_buffer(0, Some(x), 0);
        compute.set_buffer(1, Some(y), 0);
        compute.set_buffer(2, Some(output), 0);
        let threads_per_thread_group = min(
            x.length(),
            self.dist_squared_fun.max_total_threads_per_threadgroup(),
        );
        compute.dispatch_threads(
            MTLSize::new(x.length(), 1, 1),
            MTLSize::new(threads_per_thread_group, 1, 1),
        );
        compute.end_encoding();
    }

    pub fn neg(&self, x: &Buffer, commands: &CommandBufferRef) {
        let compute = commands.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&self.neg_fun);
        compute.set_buffer(0, Some(x), 0);
        let threads_per_thread_group =
            min(x.length(), self.neg_fun.max_total_threads_per_threadgroup());
        compute.dispatch_threads(
            MTLSize::new(x.length(), 1, 1),
            MTLSize::new(threads_per_thread_group, 1, 1),
        );
        compute.end_encoding();
    }

    pub fn sparse_matrix_vector_prod(
        &self,
        c: &Buffer,
        row_offsets: &Buffer,
        col_indices: &Buffer,
        values: &Buffer,
        b: &Buffer,
        commands: &CommandBufferRef,
    ) {
        let compute = commands.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&self.sparse_matrix_vector_prod_fun);
        compute.set_buffer(0, Some(c), 0);
        compute.set_buffer(1, Some(row_offsets), 0);
        compute.set_buffer(2, Some(col_indices), 0);
        compute.set_buffer(3, Some(values), 0);
        compute.set_buffer(4, Some(b), 0);
        let threads_per_thread_group = min(
            c.length(),
            self.sparse_matrix_vector_prod_fun
                .max_total_threads_per_threadgroup(),
        );
        compute.dispatch_threads(
            MTLSize::new(c.length(), 1, 1),
            MTLSize::new(threads_per_thread_group, 1, 1),
        );
        compute.end_encoding();
    }

    pub fn mpmmv_full(
        &self,
        beta: f16,
        c: &Buffer,
        alpha: f16,
        row_offsets: &Buffer,
        col_indices: &Buffer,
        values: &Buffer,
        b: &Buffer,
        commands: &CommandBufferRef,
    ) {
        let compute = commands.new_compute_command_encoder();

        compute.set_compute_pipeline_state(&self.sparse_matrix_vector_prod_fun);
        let beta = beta.to_le_bytes();
        let alpha = alpha.to_le_bytes();
        compute.set_bytes(0, size_of_val(&beta) as u64, beta.as_ptr() as *const c_void);
        compute.set_buffer(1, Some(c), 0);
        compute.set_bytes(
            2,
            size_of_val(&alpha) as u64,
            alpha.as_ptr() as *const c_void,
        );
        compute.set_buffer(3, Some(row_offsets), 0);
        compute.set_buffer(4, Some(col_indices), 0);
        compute.set_buffer(5, Some(values), 0);
        compute.set_buffer(6, Some(b), 0);
        let threads_per_thread_group = min(
            c.length(),
            self.mpmmv_full_fun.max_total_threads_per_threadgroup(),
        );
        compute.dispatch_threads(
            MTLSize::new(c.length(), 1, 1),
            MTLSize::new(threads_per_thread_group, 1, 1),
        );
        compute.end_encoding();
    }
}
