use crate::utils::{Config, NumericField, run_models};
use aph::{
    aph::Aph,
    formats::ft::utils::{ConstructionMethod, RoundMode},
    linalg::fields::{
        PseudoField, Round, float::CustomFloat, float64::Float64, inari_int::Interval,
        rational::Rational,
    },
    representation::bidiagonal::Bidiagonal,
};
use clap::Parser;
use log::info;
use serde_json::json;
use std::{
    io,
    path::{Path, PathBuf},
    time::Instant,
};

mod test;
mod utils;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Read model from tra file.
    #[arg(short, long, conflicts_with = "model")]
    input: Option<std::path::PathBuf>,
    /// Output file.
    #[arg(short, long, requires = "input")]
    output: Option<String>,
    /// Configuration parameters.
    #[command(flatten)]
    config: Config,
}

fn parse_file<F: PseudoField>(
    path: &Path,
    round: Round,
    reduce: bool,
    modularise: bool,
    method: ConstructionMethod,
    mode: RoundMode,
) -> Aph<F, Bidiagonal<F>> {
    let source = std::fs::read_to_string(path).unwrap();
    if path.extension().unwrap() == "tra" {
        let mut aph = aph::formats::tra::parse_tra(&source).unwrap();
        if reduce {
            aph.reduce();
        }
        aph
    } else if path.extension().unwrap() == "dft" {
        aph::formats::ft::aph_from_ft::<F>(&source, reduce, modularise, method, round, mode)
    } else {
        panic!("Only `.tra` or `.dft` formats supported.")
    }
}

/// Reads model from file.
/// It returns a triple with the size of the representation, the expected time until absorbtion and, if and output was provided, the return code of the save to file.
/// By default, it uses .tra files as output, but if specified .ctmc or .ma, it writes a Prism model.
pub fn from_file<F: PseudoField>(
    file: PathBuf,
    output: Option<String>,
    round: Round,
    reduce: bool,
    modularise: bool,
    method: ConstructionMethod,
    mode: RoundMode,
) -> (usize, String, io::Result<String>) {
    let aph = parse_file::<F>(&file, round, reduce, modularise, method, mode);
    let mttf = aph.expected_value().unwrap();
    let size = aph.size();
    match output {
        Some(file) if file.ends_with(".ctmc") => {
            (size, mttf.to_string(), aph.to_coxian().ctmc_export(&file))
        }
        Some(file) if file.ends_with(".ma") => (size, mttf.to_string(), aph.ma_export(&file)),
        Some(file) => {
            let file = file.replace("\"", "");
            (size, mttf.to_string(), aph.export_to_tra(&file))
        }
        None => (size, mttf.to_string(), Ok("".to_string())),
    }
}

#[allow(unreachable_code)]
fn main() {
    env_logger::init();
    let args = Args::parse();
    let time_start = Instant::now();
    info!(
        "Field: {:?} - Round: {:?}. Method: {:?}",
        args.config.numeric_field, args.config.round, args.config.method
    );

    let (size, mttf, output_file) = match args.input.clone() {
        Some(file) => match args.config.numeric_field {
            NumericField::Float => from_file::<CustomFloat>(
                file,
                args.output,
                args.config.round,
                args.config.reduce,
                args.config.modularise,
                args.config.method,
                args.config.mode,
            ),
            NumericField::Rational => from_file::<Rational>(
                file,
                args.output,
                args.config.round,
                args.config.reduce,
                args.config.modularise,
                args.config.method,
                args.config.mode,
            ),
            NumericField::F64 => from_file::<Float64>(
                file,
                args.output,
                args.config.round,
                args.config.reduce,
                args.config.modularise,
                args.config.method,
                args.config.mode,
            ),
            NumericField::Interval => from_file::<Interval>(
                file,
                args.output,
                args.config.round,
                args.config.reduce,
                args.config.modularise,
                args.config.method,
                args.config.mode,
            ),
        },
        None => run_models(args.config),
    };

    let elapsed = time_start.elapsed();

    println!(
        "{}",
        json!({
            "outfile": output_file.unwrap_or("".to_string()),
            "input": args.input.unwrap_or_else(|| serde_json::to_string(&args.config.model).unwrap().replace("\"", "").into()),
            "size": size,
            "mttf": mttf,
            "elapsed": format!("{:?}", elapsed.as_secs_f64())
        })
    )
}
