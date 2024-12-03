use std::{path::Path, time::Instant};

use clap::Parser;

use aph::{
    formats,
    linalg::fields::{PseudoField, Rational},
    representation::{Triangular, TriangularArray},
    Aph,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    file: std::path::PathBuf,
}

pub fn parse_file<F: PseudoField>(path: &Path) -> Aph<F, Triangular<F>> {
    let source = std::fs::read_to_string(path).unwrap();
    if path.extension().unwrap() == "tra" {
        formats::tra::parse(&source).unwrap()
    }
    else {
        formats::aut::parse(&source).unwrap()
    }
}

fn main() {
    let args = Args::parse();
    
    let time_start = Instant::now();
    let mut dist1: Aph<Rational, TriangularArray<Rational>> =
        formats::tra::parse_array::<Rational>(&std::fs::read_to_string(&args.file).unwrap())
            .unwrap();

    println!("Before the reduction, APH is:\n{}", dist1);
    dist1.reduce();
    println!("After the reduction, APH is:\n{}", dist1);
    let elapsed = time_start.elapsed();
    println!("Time Elapsed: {:?}", elapsed);
}
