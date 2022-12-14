use crate::image_format::PX_MAX;
use crate::inpaint_worker::CsrMatrixF16;
use half::f16;
use metal::*;
use std::cmp::min;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

const LIB: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/metallib/gpu.metallib"
));

pub struct GpuLib {
    device: Device,
    default_queue: CommandQueue,
    cg_init: ComputePipelineState,
    cg_step_0_reset_alpha_beta: ComputePipelineState,
    cg_step_1_norm_squared2: ComputePipelineState,
    cg_step_2_bp: ComputePipelineState,
    cg_step_3_1_dot_pbp: ComputePipelineState,
    cg_step_3_2_alpha: ComputePipelineState,
    cg_step_4_update_x: ComputePipelineState,
    cg_step_5_1_new_norm_squared2: ComputePipelineState,
    cg_step_5_2_beta: ComputePipelineState,
    cg_step_6_update_p: ComputePipelineState,
    cg_step_7_diff_squared: ComputePipelineState,
    ag_step_0_reset_grad_norm: ComputePipelineState,
    ag_step_2_1_yk1: ComputePipelineState,
    ag_step_3_bx_minus_c: ComputePipelineState,
    ag_step_4_grad_norm: ComputePipelineState,
    ag_step_5_update_x: ComputePipelineState,
    ag_step_6_update_beta: ComputePipelineState,
    ag_step_7_diff_squared: ComputePipelineState,
}

unsafe impl Send for GpuLib {}

fn init_module() -> GpuLib {
    let device = Device::system_default().expect("failed to find gpu");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_data(LIB)
        .expect("invalid metal library");

    GpuLib {
        device: device.clone(),
        default_queue: queue,
        cg_init: get_func("cg_init", &lib, &device),
        cg_step_0_reset_alpha_beta: get_func("cg_step_0_reset_alpha_beta", &lib, &device),
        cg_step_1_norm_squared2: get_func("cg_step_1_norm_squared2", &lib, &device),
        cg_step_2_bp: get_func("cg_step_2_bp", &lib, &device),
        cg_step_3_1_dot_pbp: get_func("cg_step_3_1_dot_pbp", &lib, &device),
        cg_step_3_2_alpha: get_func("cg_step_3_2_alpha", &lib, &device),
        cg_step_4_update_x: get_func("cg_step_4_update_x", &lib, &device),
        cg_step_5_1_new_norm_squared2: get_func("cg_step_5_1_new_norm_squared2", &lib, &device),
        cg_step_5_2_beta: get_func("cg_step_5_2_beta", &lib, &device),
        cg_step_6_update_p: get_func("cg_step_6_update_p", &lib, &device),
        cg_step_7_diff_squared: get_func("cg_step_7_diff_squared", &lib, &device),
        ag_step_0_reset_grad_norm: get_func("ag_step_0_reset_grad_norm", &lib, &device),
        ag_step_2_1_yk1: get_func("ag_step_2_1_yk1", &lib, &device),
        ag_step_3_bx_minus_c: get_func("ag_step_3_bx_minus_c", &lib, &device),
        ag_step_4_grad_norm: get_func("ag_step_4_grad_norm", &lib, &device),
        ag_step_5_update_x: get_func("ag_step_5_update_x", &lib, &device),
        ag_step_6_update_beta: get_func("ag_step_6_update_beta", &lib, &device),
        ag_step_7_diff_squared: get_func("ag_step_7_diff_squared", &lib, &device),
    }
}

fn get_func(name: &str, lib: &Library, device: &Device) -> ComputePipelineState {
    let func = lib
        .get_function(name, None)
        .unwrap_or_else(|_| panic!("failed to load function ({name})"));
    device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap_or_else(|_| panic!("failed to get pipeline of ({name})"))
}

impl GpuLib {
    pub fn init() -> Self {
        init_module()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_queue(&self) -> CommandQueue {
        self.device.new_command_queue()
    }

    pub fn private_buffer_u32(&self, data: &[usize]) -> (Buffer, &CommandBufferRef) {
        let buf_size = (size_of::<u32>() * data.len()) as u64;
        let buf = data.iter().map(|f| *f as u32).collect::<Vec<_>>();
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
    pub fn private_buffer_f16_from_u8(&self, data: &[u8]) -> (Buffer, &CommandBufferRef) {
        let buf_size = (size_of::<f16>() * data.len()) as u64;
        let buf = data
            .iter()
            .map(|f| f16::from_f32(*f as f32 / PX_MAX))
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
        let buf = data.iter().map(|f| f16::from_f32(*f)).collect::<Vec<_>>();
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

    pub fn copy_from_buffer(&self, src: &Buffer, dest: &Buffer, commands: &CommandBufferRef) {
        let buf_size = src.length();
        let blit_pass = commands.new_blit_command_encoder();
        blit_pass.copy_from_buffer(src, 0, dest, 0, buf_size);
        blit_pass.end_encoding();
    }

    pub fn init_ag_argument(
        &self,
        b_mat: &Arc<Mutex<CsrMatrixF16>>,
        c: Buffer,
        layer: Buffer,
        mu: f32,
        x: Buffer,
    ) -> (Buffer, [Buffer; 15]) {
        let arg_x = ArgumentDescriptor::new();
        arg_x.set_data_type(MTLDataType::Pointer);
        arg_x.set_index(0);
        let arg_x_tmp = ArgumentDescriptor::new();
        arg_x_tmp.set_data_type(MTLDataType::Pointer);
        arg_x_tmp.set_index(1);
        let arg_x_old = ArgumentDescriptor::new();
        arg_x_old.set_data_type(MTLDataType::Pointer);
        arg_x_old.set_index(2);
        let arg_y = ArgumentDescriptor::new();
        arg_y.set_data_type(MTLDataType::Pointer);
        arg_y.set_index(3);

        let arg_grad_norm = ArgumentDescriptor::new();
        arg_grad_norm.set_data_type(MTLDataType::Pointer);
        arg_grad_norm.set_index(4);
        let arg_dot = ArgumentDescriptor::new();
        arg_dot.set_data_type(MTLDataType::Pointer);
        arg_dot.set_index(5);
        let arg_diff_squared = ArgumentDescriptor::new();
        arg_diff_squared.set_data_type(MTLDataType::Pointer);
        arg_diff_squared.set_index(6);

        let arg_alpha = ArgumentDescriptor::new();
        arg_alpha.set_data_type(MTLDataType::Pointer);
        arg_alpha.set_index(7);
        let arg_beta = ArgumentDescriptor::new();
        arg_beta.set_data_type(MTLDataType::Pointer);
        arg_beta.set_index(8);
        let arg_t = ArgumentDescriptor::new();
        arg_t.set_data_type(MTLDataType::Pointer);
        arg_t.set_index(9);

        let arg_c = ArgumentDescriptor::new();
        arg_c.set_data_type(MTLDataType::Pointer);
        arg_c.set_index(10);
        let arg_row_offsets = ArgumentDescriptor::new();
        arg_row_offsets.set_data_type(MTLDataType::Pointer);
        arg_row_offsets.set_index(11);
        let arg_col_indices = ArgumentDescriptor::new();
        arg_col_indices.set_data_type(MTLDataType::Pointer);
        arg_col_indices.set_index(12);
        let arg_values = ArgumentDescriptor::new();
        arg_values.set_data_type(MTLDataType::Pointer);
        arg_values.set_index(13);
        let arg_original = ArgumentDescriptor::new();
        arg_original.set_data_type(MTLDataType::Pointer);
        arg_original.set_index(14);

        let encoder = self.device.new_argument_encoder(Array::from_slice(&[
            arg_x,
            arg_x_tmp,
            arg_x_old,
            arg_y,
            arg_grad_norm,
            arg_dot,
            arg_diff_squared,
            arg_alpha,
            arg_beta,
            arg_t,
            arg_c,
            arg_row_offsets,
            arg_col_indices,
            arg_values,
            arg_original,
        ]));

        let argument_buffer = self.device.new_buffer(
            encoder.encoded_length(),
            MTLResourceOptions::StorageModeShared,
        );

        let y = self
            .device
            .new_buffer(x.length(), MTLResourceOptions::StorageModePrivate);
        let x_tmp = self
            .device
            .new_buffer(x.length(), MTLResourceOptions::StorageModePrivate);
        let x_old = self
            .device
            .new_buffer(x.length(), MTLResourceOptions::StorageModePrivate);
        let grad_norm_squared = self.device.new_buffer_with_data(
            0.0f32.to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let dot = self.device.new_buffer_with_data(
            0.0f32.to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let diff_squared = self.device.new_buffer_with_data(
            0.0f32.to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let alpha = self.device.new_buffer_with_data(
            f16::from_f32(1.0 / (1.0 + 8.0 * mu)).to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let beta = self.device.new_buffer_with_data(
            f16::from_f32(0.0).to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let t = self.device.new_buffer_with_data(
            f16::from_f32(1.0).to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        encoder.set_argument_buffer(&argument_buffer, 0);
        let data = {
            let b_mat = b_mat.lock().unwrap();
            [
                x,                         // [[id(0)]]
                x_tmp,                     // [[id(1)]]
                x_old,                     // [[id(2)]]
                y,                         // [[id(3)]]
                grad_norm_squared,         // [[id(4)]]
                dot,                       // [[id(5)]]
                diff_squared,              // [[id(6)]]
                alpha,                     // [[id(7)]]
                beta,                      // [[id(8)]]
                t,                         // [[id(9)]]
                c,                         // [[id(10)]]
                b_mat.row_offsets.clone(), // [[id(11)]]
                b_mat.col_indices.clone(), // [[id(12)]]
                b_mat.values.clone(),      // [[id(13)]]
                layer,                     // [[id(14)]]
            ]
        };
        encoder.set_buffers(
            0,
            data.iter()
                .map(|i| i.as_ref())
                .collect::<Vec<_>>()
                .as_slice(),
            &[0; 15],
        );
        (argument_buffer, data)
    }

    pub fn init_cg_argument(
        &self,
        b_mat: &Arc<Mutex<CsrMatrixF16>>,
        c: Buffer,
        layer: Buffer,
        x: Buffer,
    ) -> (Buffer, [Buffer; 14]) {
        let arg_r_new_norm_squared = ArgumentDescriptor::new();
        arg_r_new_norm_squared.set_data_type(MTLDataType::Pointer);
        arg_r_new_norm_squared.set_index(0);
        let arg_r_norm_squared = ArgumentDescriptor::new();
        arg_r_norm_squared.set_data_type(MTLDataType::Pointer);
        arg_r_norm_squared.set_index(1);
        let arg_dot = ArgumentDescriptor::new();
        arg_dot.set_data_type(MTLDataType::Pointer);
        arg_dot.set_index(2);
        let arg_diff_squared = ArgumentDescriptor::new();
        arg_diff_squared.set_data_type(MTLDataType::Pointer);
        arg_diff_squared.set_index(3);

        let arg_alpha = ArgumentDescriptor::new();
        arg_alpha.set_data_type(MTLDataType::Pointer);
        arg_alpha.set_index(4);
        let arg_beta = ArgumentDescriptor::new();
        arg_beta.set_data_type(MTLDataType::Pointer);
        arg_beta.set_index(5);

        let arg_x = ArgumentDescriptor::new();
        arg_x.set_data_type(MTLDataType::Pointer);
        arg_x.set_index(6);
        let arg_bp = ArgumentDescriptor::new();
        arg_bp.set_data_type(MTLDataType::Pointer);
        arg_bp.set_index(7);
        let arg_p = ArgumentDescriptor::new();
        arg_p.set_data_type(MTLDataType::Pointer);
        arg_p.set_index(8);
        let arg_r = ArgumentDescriptor::new();
        arg_r.set_data_type(MTLDataType::Pointer);
        arg_r.set_index(9);
        let arg_row_offsets = ArgumentDescriptor::new();
        arg_row_offsets.set_data_type(MTLDataType::Pointer);
        arg_row_offsets.set_index(10);
        let arg_col_indices = ArgumentDescriptor::new();
        arg_col_indices.set_data_type(MTLDataType::Pointer);
        arg_col_indices.set_index(11);
        let arg_values = ArgumentDescriptor::new();
        arg_values.set_data_type(MTLDataType::Pointer);
        arg_values.set_index(12);
        let arg_original = ArgumentDescriptor::new();
        arg_original.set_data_type(MTLDataType::Pointer);
        arg_original.set_index(13);

        let encoder = self.device.new_argument_encoder(Array::from_slice(&[
            arg_r_new_norm_squared,
            arg_r_norm_squared,
            arg_dot,
            arg_diff_squared,
            arg_alpha,
            arg_beta,
            arg_x,
            arg_bp,
            arg_p,
            arg_r,
            arg_row_offsets,
            arg_col_indices,
            arg_values,
            arg_original,
        ]));

        let argument_buffer = self.device.new_buffer(
            encoder.encoded_length(),
            MTLResourceOptions::StorageModeShared,
        );

        let r_new_norm_squared = self.device.new_buffer_with_data(
            0.0f32.to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let r_norm_squared = self.device.new_buffer_with_data(
            0.0f32.to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let dot = self.device.new_buffer_with_data(
            0.0f32.to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let diff_squared = self.device.new_buffer_with_data(
            0.0f32.to_le_bytes().as_ptr() as *const c_void,
            size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let alpha = self.device.new_buffer_with_data(
            f16::from_f32(0.0).to_le_bytes().as_ptr() as *const c_void,
            size_of::<f16>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let beta = self.device.new_buffer_with_data(
            f16::from_f32(0.0).to_le_bytes().as_ptr() as *const c_void,
            size_of::<f16>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let bp = self
            .device
            .new_buffer(c.length(), MTLResourceOptions::StorageModePrivate);
        let p = self
            .device
            .new_buffer(c.length(), MTLResourceOptions::StorageModePrivate);

        encoder.set_argument_buffer(&argument_buffer, 0);
        let data = {
            let b_mat = b_mat.lock().unwrap();
            [
                r_new_norm_squared,        // [[id(0)]]
                r_norm_squared,            // [[id(1)]]
                dot,                       // [[id(2)]]
                diff_squared,              // [[id(3)]]
                alpha,                     // [[id(4)]]
                beta,                      // [[id(5)]]
                x,                         // [[id(6)]]
                bp,                        // [[id(7)]]
                p,                         // [[id(8)]]
                c,                         // [[id(9)]]
                b_mat.row_offsets.clone(), // [[id(10)]]
                b_mat.col_indices.clone(), // [[id(11)]]
                b_mat.values.clone(),      // [[id(12)]]
                layer,                     // [[id(13)]]
            ]
        };
        encoder.set_buffers(
            0,
            data.iter()
                .map(|i| i.as_ref())
                .collect::<Vec<_>>()
                .as_slice(),
            &[0; 14],
        );
        let cmd = self.default_queue.new_command_buffer();
        let compute = cmd.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&self.cg_init);
        compute.set_buffer(0, Some(&argument_buffer), 0);
        compute.use_resource(&data[10], MTLResourceUsage::Read);
        compute.use_resource(&data[11], MTLResourceUsage::Read);
        compute.use_resource(&data[12], MTLResourceUsage::Read);
        compute.use_resource(&data[6], MTLResourceUsage::Read);
        compute.use_resource(&data[8], MTLResourceUsage::Write);
        compute.use_resource(&data[9], MTLResourceUsage::Write);
        (argument_buffer, data)
    }

    // return r_new_norm_squared and diff_squared on each round
    pub fn iter_cg<'a>(
        &'a self,
        arg: &Buffer,
        data: &[Buffer; 14],
        queue: &'a CommandQueue,
        size: u64,
    ) -> &'a CommandBufferRef {
        let set_arg = |encoder: &ComputeCommandEncoderRef| {
            encoder.set_buffer(0, Some(arg), 0);
            encoder.use_resources(
                &[
                    &data[0], &data[1], &data[2], &data[3], &data[4], &data[5], &data[6], &data[7],
                    &data[8], &data[9],
                ],
                MTLResourceUsage::Write | MTLResourceUsage::Read,
            );
            encoder.use_resources(
                &[&data[10], &data[11], &data[12], &data[13]],
                MTLResourceUsage::Read,
            );
            encoder.end_encoding();
        };

        let command = queue.new_command_buffer();
        let step_0 = command.new_compute_command_encoder();
        step_0.set_compute_pipeline_state(&self.cg_step_0_reset_alpha_beta);
        step_0.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        set_arg(step_0);

        let step_1 = command.new_compute_command_encoder();
        step_1.set_compute_pipeline_state(&self.cg_step_1_norm_squared2);
        let max_threads = self
            .cg_step_1_norm_squared2
            .max_total_threads_per_threadgroup();
        step_1.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_1);

        let step_2 = command.new_compute_command_encoder();
        step_2.set_compute_pipeline_state(&self.cg_step_2_bp);
        let max_threads = self.cg_step_2_bp.max_total_threads_per_threadgroup();
        step_2.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_2);

        let step_3_1 = command.new_compute_command_encoder();
        step_3_1.set_compute_pipeline_state(&self.cg_step_3_1_dot_pbp);
        let max_threads = self.cg_step_3_1_dot_pbp.max_total_threads_per_threadgroup();
        step_3_1.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_3_1);

        let step_3_2 = command.new_compute_command_encoder();
        step_3_2.set_compute_pipeline_state(&self.cg_step_3_2_alpha);
        step_3_2.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        set_arg(step_3_2);

        let step_4 = command.new_compute_command_encoder();
        step_4.set_compute_pipeline_state(&self.cg_step_4_update_x);
        let max_threads = self.cg_step_4_update_x.max_total_threads_per_threadgroup();
        step_4.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_4);

        let step_5_1 = command.new_compute_command_encoder();
        step_5_1.set_compute_pipeline_state(&self.cg_step_5_1_new_norm_squared2);
        let max_threads = self
            .cg_step_5_1_new_norm_squared2
            .max_total_threads_per_threadgroup();
        step_5_1.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_5_1);

        let step_5_2 = command.new_compute_command_encoder();
        step_5_2.set_compute_pipeline_state(&self.cg_step_5_2_beta);
        step_5_2.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        set_arg(step_5_2);

        let step_6 = command.new_compute_command_encoder();
        step_6.set_compute_pipeline_state(&self.cg_step_6_update_p);
        let max_threads = self.cg_step_6_update_p.max_total_threads_per_threadgroup();
        step_6.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_6);

        let step_7 = command.new_compute_command_encoder();
        step_7.set_compute_pipeline_state(&self.cg_step_7_diff_squared);
        let max_threads = self
            .cg_step_7_diff_squared
            .max_total_threads_per_threadgroup();
        step_7.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_7);

        command
    }

    pub fn iter_ag<'a>(
        &'a self,
        arg: &Buffer,
        data: &[Buffer; 15],
        queue: &'a CommandQueue,
        size: u64,
    ) -> &'a CommandBufferRef {
        let set_arg = |encoder: &ComputeCommandEncoderRef| {
            encoder.set_buffer(0, Some(arg), 0);
            encoder.use_resources(
                &[
                    &data[0], &data[1], &data[2], &data[3], &data[4], &data[5], &data[6], &data[8],
                    &data[9],
                ],
                MTLResourceUsage::Write | MTLResourceUsage::Read,
            );
            encoder.use_resources(
                &[
                    &data[7], &data[10], &data[11], &data[12], &data[13], &data[14],
                ],
                MTLResourceUsage::Read,
            );
            encoder.end_encoding();
        };

        let command = queue.new_command_buffer();

        let step_0 = command.new_compute_command_encoder();
        step_0.set_compute_pipeline_state(&self.ag_step_0_reset_grad_norm);
        step_0.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        set_arg(step_0);

        // step 1: x_tmp <- x
        self.copy_from_buffer(&data[0], &data[1], command);

        let step_2 = command.new_compute_command_encoder();
        step_2.set_compute_pipeline_state(&self.ag_step_2_1_yk1);
        let max_threads = self.ag_step_2_1_yk1.max_total_threads_per_threadgroup();
        step_2.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_2);

        // step_2_2: y <- x
        self.copy_from_buffer(&data[0], &data[3], command);

        let step_3 = command.new_compute_command_encoder();
        step_3.set_compute_pipeline_state(&self.ag_step_3_bx_minus_c);
        let max_threads = self
            .ag_step_3_bx_minus_c
            .max_total_threads_per_threadgroup();
        step_3.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_3);

        let step_4 = command.new_compute_command_encoder();
        step_4.set_compute_pipeline_state(&self.ag_step_4_grad_norm);
        let max_threads = self.ag_step_4_grad_norm.max_total_threads_per_threadgroup();
        step_4.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_4);

        let step_5 = command.new_compute_command_encoder();
        step_5.set_compute_pipeline_state(&self.ag_step_5_update_x);
        let max_threads = self.ag_step_5_update_x.max_total_threads_per_threadgroup();
        step_5.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_5);

        let step_6 = command.new_compute_command_encoder();
        step_6.set_compute_pipeline_state(&self.ag_step_6_update_beta);
        step_6.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        set_arg(step_6);

        let step_7 = command.new_compute_command_encoder();
        step_7.set_compute_pipeline_state(&self.ag_step_7_diff_squared);
        let max_threads = self
            .ag_step_7_diff_squared
            .max_total_threads_per_threadgroup();
        step_7.dispatch_threads(
            MTLSize::new(size, 1, 1),
            MTLSize::new(min(max_threads, size), 1, 1),
        );
        set_arg(step_7);

        command
    }
}
