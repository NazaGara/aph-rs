use std::{path::Path, time::Instant};

use clap::Parser;

use aph::{
    formats,
    linalg::fields::{PseudoField, Rational},
    representation::{Triangular, TriangularArray},
    Aph,
};

// use log::info;
// use memory_stats::memory_stats;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    file: std::path::PathBuf,
}

pub fn parse_file<F: PseudoField>(path: &Path) -> Aph<F, Triangular<F>> {
    let source = std::fs::read_to_string(path).unwrap();
    if path.extension().unwrap() == "tra" {
        formats::tra::parse(&source).unwrap()
    } else {
        formats::aut::parse(&source).unwrap()
    }
}

pub fn parse_file_array<F: PseudoField>(path: &Path) -> Aph<F, TriangularArray<F>> {
    let source = std::fs::read_to_string(path).unwrap();
    if path.extension().unwrap() == "tra" {
        formats::tra::parse_array(&source).unwrap()
    } else {
        panic!("Only .tar format supported for representation.")
    }
}

fn main() {
    // std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    let args = Args::parse();
    let _time_start = Instant::now();
    let _dist1 = parse_file_array::<Rational>(&args.file);


    println!("Before the reduction, APH size: {}", _dist1.size());
    let mut bidi1 = _dist1.spa();
    bidi1.reduce();
    let _elapsed = _time_start.elapsed();
    println!("After the reduction, APH size: {}. Elapsed: {:?}", bidi1.size(), _elapsed);

    


    
}
