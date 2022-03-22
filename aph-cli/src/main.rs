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

fn main() {
    let args = Args::parse();

    let dist = parse_file::<Float64>(&args.file);

    let bidiagonal = dist.to_bidiagonal();

    println!("{bidiagonal:?}");
}
