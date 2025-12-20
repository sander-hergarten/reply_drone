mod image_processing;
mod mean;

use std::{cell::RefCell, fs, path::PathBuf, sync::Mutex};

use hashbrown::HashMap;
use image::{DynamicImage, ImageReader};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{image_processing::WgpuProcessor, mean::MeanCalculator};

const IMAGE_SIZE: (usize, usize) = (640, 720);
const INTERPOLATION_STEPS: usize = 10;
const PATH_TO_FRAMES: &str = "/Users/sanderhergarten/Documents/programming/reply_drone/crates/dataset_generation/captures/simple/frames";
const PATH_TO_MASKS: &str = "/Users/sanderhergarten/Documents/programming/reply_drone/crates/dataset_generation/captures/simple/mask";
const CHUNK_SIZE: usize = 500;

struct Progress {
    paths: Vec<PathBuf>,
    processed: RefCell<usize>,
}

impl Progress {
    fn new(directory_containing_images: PathBuf) -> Option<Self> {
        let paths = fs::read_dir(directory_containing_images)
            .unwrap()
            .map(|entry| entry.unwrap())
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        Some(Self {
            paths,
            processed: RefCell::new(0),
        })
    }
    fn from_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            processed: RefCell::new(0),
        }
    }

    fn get_next_paths(&self, batch_size: usize) -> Vec<PathBuf> {
        let slice = &self.paths[*self.processed.borrow()..*self.processed.borrow() + batch_size];
        (*self.processed.borrow_mut()) += batch_size;
        slice.to_vec()
    }

    fn get_chunked_paths(&self) -> Option<Vec<PathBuf>> {
        let chunk_size = self
            .paths
            .len()
            .min(*self.processed.borrow() + CHUNK_SIZE)
            .checked_sub(*self.processed.borrow())
            .expect("tried out of bounds access");

        println!("{}", *self.processed.borrow());
        if chunk_size == 0 {
            None
        } else {
            Some(self.get_next_paths(chunk_size))
        }
    }
}

fn main() {
    let mask_paths = balance_dataset();
    let processor = pollster::block_on(WgpuProcessor::new(0.2));

    let frame_progress = Progress::from_paths(frame_path_from_mask_path(&mask_paths));
    let mask_progress = Progress::from_paths(mask_paths);
    while let Some(masks_dyn) = load_batch_into_memory(&mask_progress) {
        let images = load_batch_into_memory(&frame_progress)
            .unwrap()
            .into_iter()
            .map(|(_, im)| im.to_rgb8())
            .collect_vec();

        let masks = masks_dyn
            .into_iter()
            .map(|(_, im)| im.to_luma8())
            .collect_vec();

        let results = processor.process_batch(&images);
    }

    //
    // let mut processor = pollster::block_on(WgpuProcessor::new(0.2));
    //
    // // 3. Process All

    //
    // // 4. Save
    // for (i, res) in results.iter().enumerate() {
    //     res.save(format!("cool/result_{}.png", i)).unwrap();
    // }
    //     let schema = Schema::new(vec![
    //         Field::new("image", DataType::Binary, false),
    //         Field::new("mask", DataType::Binary, false),
    //     ]);
    //     let schema_ref = Arc::new(schema);
    //
    //     // 2. Prepare Data Builders
    //     let mut image_builder = BinaryBuilder::new();
    //     let mut mask_builder = BinaryBuilder::new();
    //
    //     let image_path = "example.jpg"; // specific path to your image
    //     let mut file = File::open(image_path)?;
    //     let mut buffer = Vec::new();
    //     file.read_to_end(&mut buffer)?;
    //
    //     // Append data to Arrow arrays
    //     image_builder.append_value(&buffer);
    //     label_builder.append_value("cat");
    //
    //     let progress = Progress::new(PathBuf::from("/Users/sanderhergarten/Documents/programming/reply_drone/crates/dataset_generation/captures/simple/frames")).expect("directory wasnt able to open");
    //
    //     let images = load_batch_into_memory(100, progress);
    //     let data = RecordBatch::try_from_iter(batch_to_data(images)).unwrap();
    //     // let writer = ArrowWriter::try_new(, data.schema(), None)
}

fn load_batch_into_memory(progress: &Progress) -> Option<Vec<(PathBuf, DynamicImage)>> {
    let paths = progress.get_chunked_paths()?;

    Some(
        paths
            .into_par_iter()
            .map(|path| {
                (
                    path.clone(),
                    ImageReader::open(path).unwrap().decode().unwrap(),
                )
            })
            .collect(),
    )
}

fn balance_dataset() -> Vec<PathBuf> {
    let mut calc = pollster::block_on(MeanCalculator::new());
    let mut means_collector: Vec<(f32, PathBuf)> = Vec::default();

    let mask_progress =
        Progress::new(PathBuf::from(PATH_TO_MASKS)).expect("directory wasnt able to open");

    while let Some(batch) = load_batch_into_memory(&mask_progress) {
        let (paths, images): (Vec<_>, Vec<_>) = batch
            .into_iter()
            .map(|(path, dynamic_image)| (path, dynamic_image.to_luma8()))
            .unzip();

        let means = calc.calculate_batch(&images);
        means_collector.extend(means.into_iter().zip(paths).collect_vec());
    }

    let target_samples = means_collector.len().div_ceil(10);
    let (_, even_paths): (Vec<f32>, Vec<PathBuf>) =
        distribute_evenly(means_collector, target_samples)
            .into_iter()
            .unzip();
    even_paths
}

fn distribute_evenly(mut items: Vec<(f32, PathBuf)>, target_count: usize) -> Vec<(f32, PathBuf)> {
    // 1. Edge Case: If we have fewer items than we want, keep them all
    if items.len() <= target_count {
        return items;
    }

    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let min_val = items.first().unwrap().0;
    let max_val = items.last().unwrap().0;
    let range = max_val - min_val;

    if range.abs() < f32::EPSILON {
        return items.into_iter().take(target_count).collect();
    }

    let mut result = Vec::with_capacity(target_count);

    let mut cursor = 0;

    for i in 0..target_count {
        let fraction = i as f32 / (target_count - 1) as f32;
        let target_val = min_val + (range * fraction);

        while cursor < items.len() - 1 && items[cursor].0 < target_val {
            cursor += 1;
        }

        let closest_idx = if cursor == 0 {
            0
        } else {
            let curr_dist = (items[cursor].0 - target_val).abs();
            let prev_dist = (items[cursor - 1].0 - target_val).abs();

            if prev_dist < curr_dist {
                cursor - 1
            } else {
                cursor
            }
        };

        result.push(items[closest_idx].clone());
    }

    result.dedup_by(|a, b| a.1 == b.1);

    result
}

fn frame_path_from_mask_path(mask_paths: &[PathBuf]) -> Vec<PathBuf> {
    mask_paths
        .iter()
        .map(|k| PathBuf::from(PATH_TO_FRAMES).join(k.file_name().unwrap()))
        .collect_vec()
}
