use std::path::Path;

use clap::Parser;

use aph::{
    formats,
    linalg::{
        fields::{Float64, PseudoField},
        Vector,
    },
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

    // let dist = parse_file::<Float64>(&args.file);

    // let bidiagonal = dist.to_bidiagonal();
    // println!("{:?}", bidiagonal);

    let dist =
        formats::tra::parse_array::<Float64>(&std::fs::read_to_string(args.file).unwrap()).unwrap();

    let bidi = dist.to_bidiagonal();

    println!("{:?}\n\n", bidi);

    let m1 = bidi.repr().clone();
    let m2 = bidi.repr().clone();

    let m = Aph::new(
        Vector::one_and_zeros(0, m1.0.len() * m2.0.len()),
        m1.kron_sum(&m2),
    );

    println!("{:?}", m);

    // let bidiagonal = m.to_bidiagonal();

    // println!("{:?}", bidiagonal);

    // let m1 = dist.repr().clone();
    // let m2 = dist.repr().clone();

    // let m = Aph::new(
    //     Vector::one_and_zeros(0, m1.size * m2.size),
    //     m1.kron_product(&m2),
    // );

    // println!("{:?}", m);

    // let bidiagonal = m.to_bidiagonal();

    // println!("{:?}", bidiagonal);
}
