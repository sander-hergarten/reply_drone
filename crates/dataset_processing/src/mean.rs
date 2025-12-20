use image::GrayImage;
use std::borrow::Cow;

pub struct MeanCalculator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    // Cached resources (Texture/Buffers) to avoid re-allocating per image
    resources: Option<MeanResources>,
}

struct MeanResources {
    width: u32,
    height: u32,
    input_texture: wgpu::Texture,
    partial_sum_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    output_size_bytes: u64,
}

impl MeanCalculator {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        // 1. Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mean Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("mean.wgsl"))),
        });

        // 2. Layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Input Texture (ReadOnly)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output Buffer (ReadWrite)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            resources: None,
        }
    }

    /// Calculates the mean luma (0.0 - 1.0) for a list of images
    pub fn calculate_batch(&mut self, images: &[GrayImage]) -> Vec<f32> {
        let mut results = Vec::with_capacity(images.len());

        for (i, img) in images.iter().enumerate() {
            let (width, height) = img.dimensions();

            // --- 1. Resource Management ---
            // Re-allocate ONLY if dimensions change or it's the first run
            let needs_alloc = match &self.resources {
                Some(res) => res.width != width || res.height != height,
                None => true,
            };

            if needs_alloc {
                self.allocate_resources(width, height);
            }

            let res = self.resources.as_ref().unwrap();

            // --- 2. Upload Image ---
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &res.input_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                img,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            // --- 3. Compute (Reduction) ---
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &res.bind_group, &[]);
                // Dispatch 1 threadgroup per 16x16 block
                cpass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
            }

            // Copy partial sums to staging buffer for CPU reading
            encoder.copy_buffer_to_buffer(
                &res.partial_sum_buffer,
                0,
                &res.staging_buffer,
                0,
                res.output_size_bytes,
            );
            self.queue.submit(Some(encoder.finish()));

            // --- 4. Readback & Final Sum ---
            let buffer_slice = res.staging_buffer.slice(..);
            let (sender, mut receiver) = futures::channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            loop {
                self.device.poll(wgpu::PollType::Poll).unwrap();
                if let Ok(Some(_)) = receiver.try_recv() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }

            let data = buffer_slice.get_mapped_range();
            let partial_sums: &[f32] = bytemuck::cast_slice(&data);

            // Sum the partial results on CPU (very fast, as it's only ~0.003% of original pixels)
            let total_sum: f32 = partial_sums.iter().sum();

            drop(data);
            res.staging_buffer.unmap();

            // Calculate Mean: Total Sum / Total Pixels
            results.push(total_sum / (width as f32 * height as f32));
        }
        results
    }

    fn allocate_resources(&mut self, width: u32, height: u32) {
        // Calculate number of workgroups (blocks)
        let wg_x = (width + 15) / 16;
        let wg_y = (height + 15) / 16;
        let num_elements = (wg_x * wg_y) as u64;
        let output_size_bytes = num_elements * 4; // f32 is 4 bytes

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let input_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm, // GPU reads as 0.0-1.0 float automatically
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Buffer to store the partial result from each workgroup
        let partial_sum_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Partial Sums"),
            size: output_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Buffer to read back to CPU
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: output_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &input_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partial_sum_buffer.as_entire_binding(),
                },
            ],
        });

        self.resources = Some(MeanResources {
            width,
            height,
            input_texture,
            partial_sum_buffer,
            staging_buffer,
            bind_group,
            output_size_bytes,
        });
    }
}
