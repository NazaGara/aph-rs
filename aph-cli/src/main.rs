use std::path::Path;

use clap::Parser;

use aph::{
    formats,
    linalg::fields::{Float64, PseudoField},
    Aph, Triangular,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    file1: std::path::PathBuf,
    file2: std::path::PathBuf,
}

pub fn parse_file<F: PseudoField>(path: &Path) -> Aph<F, Triangular<F>> {
    let source = std::fs::read_to_string(path).unwrap();
    if path.extension().unwrap() == "tra" {
        formats::tra::parse(&source).unwrap()
    } else {
        formats::aut::parse(&source).unwrap()
    }
}

fn main() {
    let args = Args::parse();

    let dist1: Aph<Float64, aph::TriangularArray<Float64>> =
        formats::tra::parse_array::<Float64>(&std::fs::read_to_string(&args.file1).unwrap())
            .unwrap();

    let dist2: Aph<Float64, aph::TriangularArray<Float64>> =
        formats::tra::parse_array::<Float64>(&std::fs::read_to_string(args.file2).unwrap())
            .unwrap();

    let bidi1 = dist1.to_bidiagonal();
    let bidi2 = dist1.to_bidiagonal();

    println!("{} vs {} \n {} vs {}", dist1, bidi1, dist2, bidi2);
}
