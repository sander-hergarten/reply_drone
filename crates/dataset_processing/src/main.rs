mod image_processing;

use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use arrow::array::{ArrayRef, FixedSizeListArray, RecordBatch};
use hashbrown::HashMap;
use image::{DynamicImage, ImageReader, RgbImage};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const IMAGE_SIZE: (usize, usize) = (640, 720);
const INTERPOLATION_STEPS: usize = 10;
const PATH_TO_FRAMES: &str = "/Users/sanderhergarten/Documents/programming/reply_drone/crates/dataset_generation/captures/simple/frames";
const PATH_TO_MASKS: &str = "/Users/sanderhergarten/Documents/programming/reply_drone/crates/dataset_generation/captures/simple/mask";
const CHUNK_SIZE: usize = 100;

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

    fn get_next_paths(&self, batch_size: usize) -> Vec<PathBuf> {
        (*self.processed.borrow_mut()) += batch_size;
        self.paths[self.processed.take() - batch_size..self.processed.take()].to_vec()
    }

    fn get_chunked_paths(&self) -> Option<Vec<PathBuf>> {
        let chunk_size = self
            .paths
            .len()
            .min(self.processed.take() + CHUNK_SIZE)
            .checked_sub(self.processed.take())
            .expect("tried out of bounds access");

        if chunk_size == 0 {
            None
        } else {
            Some(self.get_next_paths(chunk_size))
        }
    }
}

fn main() {
    pollster::block_on(image_processing::run());
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

fn process_to_parquet() {}

fn load_batch_into_memory(progress: &Progress) -> Option<Vec<DynamicImage>> {
    let paths = progress.get_chunked_paths()?;

    Some(
        paths
            .into_par_iter()
            .map(|path| ImageReader::open(path).unwrap().decode().unwrap())
            .collect(),
    )
}

fn balance_dataset() {
    let mask_mean: HashMap<PathBuf, f64> = HashMap::new();

    let mask_progress =
        Progress::new(PathBuf::from(PATH_TO_MASKS)).expect("directory wasnt able to open");

    while let Some(batch) = load_batch_into_memory(&mask_progress) {
        let images = batch
            .into_iter()
            .map(|dynamic_image| dynamic_image.to_luma8());
    }
}
