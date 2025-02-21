use std::{
    io,
    path::{Path, PathBuf},
    time::Instant,
};

use aph::{
    formats::{self},
    linalg::fields::{
        float64::Float64, float64_cf::Float64RoundCF, float64_round::Float64Round, inari_int::Interval, rational::Rational, Down, Near, PseudoField, Up
    },
    representation::{Triangular, TriangularArray},
    Aph,
};
use clap::{Parser, ValueEnum};
use log::{info, warn};
use memory_stats::memory_stats;

pub mod models;
use models::*;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Read model from tra file.
    #[arg(short, long, conflicts_with = "model")]
    input: Option<std::path::PathBuf>,
    /// Output file, writes a tra file.
    #[arg(short, long, requires = "file")]
    output: Option<String>,
    #[arg(value_enum, short, long, default_value_t=NumericField::Rational)]
    numeric_field: NumericField,
    #[arg(value_enum, short, long, default_value_t=Model::Ex37, conflicts_with="input")]
    model: Model,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum NumericField {
    Rational,
    F64,
    F64Up,
    F64Down,
    CFF64,
    CFF64Up,
    CFF64Down,
    Inari,
}

fn run(field: NumericField, model: Model) -> io::Result<()> {
    match field {
        NumericField::Rational => choose_model::<Rational>(model),
        NumericField::F64 => choose_model::<Float64>(model),
        NumericField::F64Down => choose_model::<Float64Round<Down>>(model),
        NumericField::F64Up => choose_model::<Float64Round<Up>>(model),
        NumericField::CFF64 => choose_model::<Float64RoundCF<Near>>(model),
        NumericField::CFF64Down => choose_model::<Float64RoundCF<Down>>(model),
        NumericField::CFF64Up => choose_model::<Float64RoundCF<Up>>(model),
        NumericField::Inari => choose_model::<Interval>(model),
    }
}

pub fn parse_file_tri<F: PseudoField>(path: &Path) -> Aph<F, Triangular<F>> {
    let source = std::fs::read_to_string(path).unwrap();
    if path.extension().unwrap() == "tra" {
        formats::tra::parse_tri(&source).unwrap()
    } else {
        panic!("Only .tar format supported for representation.")
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
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    let args = Args::parse();
    let (pre_physical_mem, pre_virtual_mem) = if let Some(usage) = memory_stats() {
        (usage.physical_mem, usage.virtual_mem)
    } else {
        warn!("Couldn't get the current memory usage :(");
        (0, 0)
    };
    let time_start = Instant::now();
    info!("Using [{:?}] as numerical field.", args.numeric_field);
    match args.input {
        Some(file) => {
            let _ = match args.numeric_field {
                NumericField::Rational => _from_file::<Rational>(file, &args.output),
                NumericField::F64 => _from_file::<Float64>(file, &args.output),
                NumericField::F64Down => _from_file::<Float64Round<Down>>(file, &args.output),
                NumericField::F64Up => _from_file::<Float64Round<Up>>(file, &args.output),
                NumericField::CFF64 => _from_file::<Float64RoundCF<Near>>(file, &args.output),
                NumericField::CFF64Down => _from_file::<Float64RoundCF<Down>>(file, &args.output),
                NumericField::CFF64Up => _from_file::<Float64RoundCF<Up>>(file, &args.output),
                NumericField::Inari => _from_file::<Interval>(file, &args.output),
                // _ => Ok(()),
            };
        }
        None => {
            let _ = run(args.numeric_field, args.model);
        }
    }

    let elapsed = time_start.elapsed();
    let (post_physical_mem, post_virtual_mem) = if let Some(usage) = memory_stats() {
        (usage.physical_mem, usage.virtual_mem)
    } else {
        warn!("Couldn't get the current memory usage :(");
        (0, 0)
    };

    info!(
        "Elapsed: {:?}. physical mem used: {:.2} MB. virtual mem used: {:.2} MB",
        elapsed,
        (post_physical_mem - pre_physical_mem) as f64 / 1048576.0,
        (post_virtual_mem - pre_virtual_mem) as f64 / 1048576.0
    );
}

pub fn _from_file<F: PseudoField>(file: PathBuf, output: &Option<String>) -> io::Result<()> {
    let time_start = Instant::now();
    let dist = parse_file_tri::<F>(&file);
    let mut bidi = dist.spa();
    bidi.reduce();
    let elapsed = time_start.elapsed();
    match output {
        Some(file) => {
            bidi._export_to_tra(file)?;
            bidi.to_coxian().export(file)?;
        }
        None => {}
    };
    info!(
        "After the reduction, APH size: {}. Elapsed: {:?}",
        bidi.size(),
        elapsed
    );
    Ok(())
}
