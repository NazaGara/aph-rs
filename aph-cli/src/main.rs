use std::{io, path::Path, time::Instant};

use aph::{
    con_ph,
    formats,
    linalg::{
        fields::{Float64, FromRational, IntervalField, PseudoField, Rational},
        Vector,
    },
    //max_ph,
    min_ph,
    representation::{Bidiagonal, Triangular, TriangularArray},
    Aph,
};
use clap::Parser;
use memory_stats::memory_stats;

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
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    // let args = Args::parse();

    let (pre_physical_mem, pre_virtual_mem) = if let Some(usage) = memory_stats() {
        (usage.physical_mem, usage.virtual_mem)
    } else {
        panic!("Couldn't get the current memory usage :(");
    };

    let time_start = Instant::now();

    // let _ = _ex3_7_rational();
    // let _ = _ex3_7_float64();
    // let _ = _ex3_7_interval();
    let _ = _steal_exam::<Rational>();
    let elapsed = time_start.elapsed();

    let (post_physical_mem, post_virtual_mem) = if let Some(usage) = memory_stats() {
        (usage.physical_mem, usage.virtual_mem)
    } else {
        panic!("Couldn't get the current memory usage :(");
    };
    println!(
        "Elapsed: {:?}. physical mem used: {:.2} MB. virtual mem used: {:.2} MB",
        elapsed,
        (post_physical_mem - pre_physical_mem) as f64 / 1048576.0,
        (post_virtual_mem - pre_virtual_mem) as f64 / 1048576.0
    );
}

fn _ex3_7_float64() {
    let mut diag = Bidiagonal::new(7);
    diag.set(0, Float64::from_rational("-1", "1"));
    diag.set(1, Float64::from_rational("-2", "1"));
    diag.set(2, Float64::from_rational("-2", "1"));
    diag.set(3, Float64::from_rational("-3", "1"));
    diag.set(4, Float64::from_rational("-4", "1"));
    diag.set(5, Float64::from_rational("-5", "1"));
    diag.set(6, Float64::from_rational("-5", "1"));
    let ini = Vector::from(vec![
        Float64::from_rational("1", "5"),
        Float64::from_rational("1", "5"),
        Float64::from_rational("2", "5"),
        Float64::from_rational("1", "5"),
        Float64::from_rational("0", "1"),
        Float64::from_rational("0", "1"),
        Float64::from_rational("0", "1"),
    ]);
    let mut ph = Aph::new(ini, diag);
    ph.reduce();
}

fn _ex3_7_interval() {
    let mut diag: Bidiagonal<IntervalField<Float64>> = Bidiagonal::new(7);
    diag.set(0, IntervalField::from_rational("-1", "1"));
    diag.set(1, IntervalField::from_rational("-2", "1"));
    diag.set(2, IntervalField::from_rational("-2", "1"));
    diag.set(3, IntervalField::from_rational("-3", "1"));
    diag.set(4, IntervalField::from_rational("-4", "1"));
    diag.set(5, IntervalField::from_rational("-5", "1"));
    diag.set(6, IntervalField::from_rational("-5", "1"));
    let ini = Vector::from(vec![
        IntervalField::from_rational("1", "5"),
        IntervalField::from_rational("1", "5"),
        IntervalField::from_rational("2", "5"),
        IntervalField::from_rational("1", "5"),
        IntervalField::from_rational("0", "1"),
        IntervalField::from_rational("0", "1"),
        IntervalField::from_rational("0", "1"),
    ]);
    let mut ph = Aph::new(ini, diag);
    ph.reduce();
}

fn _ex3_7_rational() {
    let mut diag = Bidiagonal::new(7);
    diag.set(0, Rational::from_rational("-1", "1"));
    diag.set(1, Rational::from_rational("-2", "1"));
    diag.set(2, Rational::from_rational("-2", "1"));
    diag.set(3, Rational::from_rational("-3", "1"));
    diag.set(4, Rational::from_rational("-4", "1"));
    diag.set(5, Rational::from_rational("-5", "1"));
    diag.set(6, Rational::from_rational("-5", "1"));
    let ini = Vector::from(vec![
        Rational::from_rational("1", "5"),
        Rational::from_rational("1", "5"),
        Rational::from_rational("2", "5"),
        Rational::from_rational("1", "5"),
        Rational::from_rational("0", "1"),
        Rational::from_rational("0", "1"),
        Rational::from_rational("0", "1"),
    ]);
    let mut ph = Aph::new(ini, diag);
    ph.reduce();
}

fn _from_file(args: Args) {
    let time_start = Instant::now();
    let dist1 = parse_file_array::<Rational>(&args.file);
    let mut bidi1 = dist1.spa();

    println!("{}", bidi1);

    bidi1.reduce();
    let elapsed = time_start.elapsed();
    println!(
        "After the reduction, APH size: {}. Elapsed: {:?}",
        bidi1.size(),
        elapsed
    );
}

fn _steal_exam<F: PseudoField>() -> io::Result<()> {
    // -------------------------------------------------------------    
    println!("Social Interaction section");
    let prep_bribe = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-4", "1");
    let prep_threa = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-3", "1");
    let prepare = min_ph(&prep_bribe, &prep_threa).spa();

    let approach = Aph::<F, Bidiagonal<F>>::_new_exp("-4", "1");
    let mut social_int = con_ph(&prepare, &approach).spa();

    println!("social_int: {:?}", social_int.size());
    social_int.reduce();
    println!("social_int: {:?}", social_int.size());

    // -------------------------------------------------------------
    // println!("Hacking section");
    // let hack = Aph::<F, Bidiagonal<F>>::_new_erl(6, "-4", "1");
    // let keylogger = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    // let guessing = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "2");
    // let crack_firewall = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    // let penetrate_repo = Aph::<F, Bidiagonal<F>>::_new_exp("-4", "1");

    // let mut acq_pass = min_ph(&keylogger, &guessing).spa();
    // acq_pass.reduce();

    // let mut mailbox = min_ph(&hack, &acq_pass).spa();
    // let mut repo = con_ph(&crack_firewall, &penetrate_repo).spa();

    // mailbox.reduce();
    // repo.reduce();

    // let hacking = min_ph(&mailbox, &repo);
    // println!("hacking: {:?}", hacking.size());
    // // hacking.dump_to_tra("hacking.tra")?;
    // let mut hacking_bidi = hacking.spa();
    // hacking_bidi.reduce();
    // println!("hacking: {:?}", hacking_bidi.size());
    // hacking_bidi.dump_to_tra("reduced.tra")?;

    // -------------------------------------------------------------
    println!("Steal Hardcopy section");
    let loc_office = Aph::<F, Bidiagonal<F>>::_new_exp("-10", "1");
    let steal_key = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-6", "1");
    let break_into_ofi = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-2", "1");
    let find_prints = Aph::<F, Bidiagonal<F>>::_new_exp("-24", "1");

    let mut get_access = min_ph(&steal_key, &break_into_ofi).spa();
    get_access.reduce();

    let steal_copy1 = con_ph(&loc_office, &get_access).spa();
    let mut steal_copy = con_ph(&steal_copy1, &find_prints).spa();

    println!("steal_copy: {:?}", steal_copy.size());
    steal_copy.reduce();
    println!("steal_copy: {:?}", steal_copy.size());
    // -------------------------------------------------------------

    // println!("Final minimization section");
    // let mut steal_exam1 = min_ph(&social_int, &hacking).spa();
    // steal_exam1.reduce();
    // let mut steal_exam = min_ph(&steal_exam1, &steal_copy).spa();
    // steal_exam.reduce();
    // steal_exam.dump_to_tra("final")?;
    Ok(())
}
