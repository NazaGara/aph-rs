use std::{
    io,
    path::{Path, PathBuf},
    time::Instant,
};

use aph::{
    formats::{self, dft},
    linalg::fields::{
        float64::Float64, float64_round::Float64Round, inari_int::Interval, interval_field::IF,
        rational::Rational, Down, PseudoField, Up,
    },
    representation::{Triangular, TriangularArray},
    Aph,
};
use clap::{Parser, ValueEnum};
use log::{info, warn};
use memory_stats::memory_stats;
use serde::{Deserialize, Serialize};
use serde_json::json;
pub mod models;
use models::*;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Read model from tra file.
    #[arg(short, long, conflicts_with = "model")]
    input: Option<std::path::PathBuf>,
    /// Output file, writes a tra file.
    #[arg(short, long, requires = "input")]
    output: Option<String>,
    #[arg(value_enum, short, long, default_value_t=NumericField::Rational)]
    numeric_field: NumericField,
    #[arg(value_enum, short, long, default_value_t=Model::Ex37, conflicts_with="input")]
    model: Model,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum NumericField {
    Rational,
    F64,
    F64Up,
    F64Down,
    Inari,
    IntF64Sup,
    IntF64Inf,
}

#[allow(unused)]
fn run(field: NumericField, model: Model) -> (usize, usize, io::Result<()>) {
    match field {
        NumericField::Rational => choose_model::<Rational>(model, "rational"),
        NumericField::F64 => choose_model::<Float64>(model, "f64"),
        NumericField::F64Down => choose_model::<Float64Round<Down>>(model, "f64-down"),
        NumericField::F64Up => choose_model::<Float64Round<Up>>(model, "f64-up"),
        NumericField::Inari => choose_model::<Interval<Down>>(model, "inari"),
        NumericField::IntF64Sup => choose_model::<IF<Float64, Up>>(model, "int-f64-sup"),
        NumericField::IntF64Inf => choose_model::<IF<Float64, Down>>(model, "int-f64-inf"),
        _ => (0, 0, Ok(())),
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

    let (reductions, size, _result) = match args.input {
        Some(file) => match args.numeric_field {
            NumericField::Rational => _from_file::<Rational>(file, args.output),
            NumericField::F64 => _from_file::<Float64>(file, args.output),
            NumericField::F64Down => _from_file::<Float64Round<Down>>(file, args.output),
            NumericField::F64Up => _from_file::<Float64Round<Up>>(file, args.output),
            NumericField::Inari => _from_file::<Interval<Down>>(file, args.output),
            _ => (0, 0, Ok(())),
        },
        None => run(args.numeric_field, args.model),
    };

    let elapsed = time_start.elapsed();
    let (post_physical_mem, post_virtual_mem) = if let Some(usage) = memory_stats() {
        (usage.physical_mem, usage.virtual_mem)
    } else {
        warn!("Couldn't get the current memory usage :(");
        (0, 0)
    };

    println!(
        "{}",
        json!({
            "model": args.model,
            "field": args.numeric_field,
            "memory": ((post_physical_mem - pre_physical_mem) as f64 / 1048576.0,
            (post_virtual_mem - pre_virtual_mem) as f64 / 1048576.0),
            "reductions": reductions,
            "size": size,
            "elapsed": format!("{:?}", elapsed.as_secs_f64())
        })
    );
}

pub fn _from_file<F: PseudoField>(
    file: PathBuf,
    output: Option<String>,
) -> (usize, usize, io::Result<()>) {
    let time_start = Instant::now();
    let dist = parse_file_tri::<F>(&file);
    let mut bidi = dist.spa();
    let total_red = bidi.reduce();
    let elapsed = time_start.elapsed();
    info!(
        "After the reduction, APH size: {}. Elapsed: {:?}",
        bidi.size(),
        elapsed.as_secs_f64()
    );
    let size = bidi.size();
    match output {
        Some(file) => (total_red, size, bidi.to_coxian().export(&file)),
        None => (total_red, size, Ok(())),
    }
}

pub fn _from_dft_file<F: PseudoField>(
    file: PathBuf,
    _output: Option<String>,
) -> (usize, io::Result<()>) {
    let time_start = Instant::now();
    let ft = dft::FaultTree::<F>::from_file(&file);

    let (aph, total_red) = match ft.nodes.get(&ft.toplevel) {
        Some(node) => node.to_aph(&ft.nodes),
        None => panic!(),
    };
    let elapsed = time_start.elapsed();
    info!("Elapsed time: {:?}. Reductions: {}", elapsed, total_red);
    match _output {
        None => (total_red, Ok(())),
        Some(file) => (total_red, aph.to_coxian().export(&file)),
    }
}
