use std::{
    io::{self},
    path::{Path, PathBuf},
    time::Instant,
};

use aph::{
    formats,
    linalg::{
        fields::{
            float32::Float32, float64::Float64, interval_field::IntervalField, rational::Rational,
            PseudoField,
        },
        Vector,
    },
    operations::*,
    representation::{Bidiagonal, TriangularArray, Triangular},
    Aph,
};
use clap::{Parser, ValueEnum};
use log::info;
use memory_stats::memory_stats;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Read model from tra file.
    #[arg(short, long)]
    input: Option<std::path::PathBuf>,
    /// Output file, writes a tra file.
    #[arg(short, long, requires = "file")]
    output: Option<String>,
    #[arg(value_enum, long, default_value_t=NumericField::Rational)]
    numeric_field: NumericField,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum NumericField {
    Rational,
    F64,
    F32,
    IntervalF32,
    IntervalF64,
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
        panic!("Couldn't get the current memory usage :(");
    };
    let time_start = Instant::now();
    info!("Using {:?} numeric field.", args.numeric_field);
    match args.input {
        Some(file) => {
            let _ = match args.numeric_field {
                NumericField::F32 => _from_file::<Float32>(file, &args.output),
                NumericField::F64 => _from_file::<Float64>(file, &args.output),
                NumericField::IntervalF32 => {
                    _from_file::<IntervalField<Float32>>(file, &args.output)
                }
                NumericField::IntervalF64 => {
                    _from_file::<IntervalField<Float64>>(file, &args.output)
                }
                NumericField::Rational => _from_file::<Rational>(file, &args.output),
            };
        }
        None => {
            let _ = match args.numeric_field {
                NumericField::F32 => _hacking::<Float32>(),
                NumericField::F64 => _hacking::<Float64>(),
                NumericField::IntervalF32 => _hacking::<IntervalField<Float32>>(),
                NumericField::IntervalF64 => _hacking::<IntervalField<Float64>>(),
                NumericField::Rational => _hacking::<Rational>(),
            };
        }
    }

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

fn _ex3_7<F: PseudoField>() {
    let mut diag = Bidiagonal::new(7);
    diag.set(0, F::from_rational("-1", "1"));
    diag.set(1, F::from_rational("-2", "1"));
    diag.set(2, F::from_rational("-2", "1"));
    diag.set(3, F::from_rational("-3", "1"));
    diag.set(4, F::from_rational("-4", "1"));
    diag.set(5, F::from_rational("-5", "1"));
    diag.set(6, F::from_rational("-5", "1"));
    let ini = Vector::from(vec![
        F::from_rational("1", "5"),
        F::from_rational("1", "5"),
        F::from_rational("2", "5"),
        F::from_rational("1", "5"),
        F::from_rational("0", "1"),
        F::from_rational("0", "1"),
        F::from_rational("0", "1"),
    ]);
    let mut ph = Aph::new(ini, diag);
    ph.reduce();
}

fn _from_file<F: PseudoField>(file: PathBuf, output: &Option<String>) {
    let time_start = Instant::now();
    let dist = parse_file_tri::<Rational>(&file);

    let mut bidi = dist.spa();
    bidi.reduce();

    let elapsed = time_start.elapsed();
    let _ = match output {
        None => Ok(()),
        Some(file) => bidi.export_tra_with_labels(&format!("{}", file)),
    };
    println!(
        "After the reduction, APH size: {}. Elapsed: {:?}",
        bidi.size(),
        elapsed
    );
}

fn _hacking<F: PseudoField>() -> io::Result<()> {
    let hack = Aph::<F, Bidiagonal<F>>::_new_erl(6, "-4", "1");
    let keylogger = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    let guessing = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "2");
    let crack_firewall = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    let penetrate_repo = Aph::<F, Bidiagonal<F>>::_new_exp("-4", "1");

    let mut acq_pass = min_ph(&keylogger, &guessing).spa();
    acq_pass.reduce();

    let mut mailbox = min_ph(&hack, &acq_pass).spa();
    let mut repo = con_ph(&crack_firewall, &penetrate_repo).spa();

    mailbox.reduce();
    repo.reduce();

    let mut hacking = min_ph(&mailbox, &repo);
    hacking.reduce();
    println!("hacking: {:?}", hacking.size());
    hacking.export_tra_with_labels("hacking")?;
    Ok(())
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
    let hack = Aph::<F, Bidiagonal<F>>::_new_erl(6, "-4", "1");
    let keylogger = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    let guessing = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "2");
    let crack_firewall = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    let penetrate_repo = Aph::<F, Bidiagonal<F>>::_new_exp("-4", "1");

    let mut acq_pass = min_ph(&keylogger, &guessing).spa();
    acq_pass.reduce();

    let mut mailbox = min_ph(&hack, &acq_pass).spa();
    let mut repo = con_ph(&crack_firewall, &penetrate_repo).spa();

    mailbox.reduce();
    repo.reduce();

    let mut hacking = min_ph(&mailbox, &repo);
    // println!("hacking: {:?}", hacking.size());
    hacking.reduce();
    // println!("hacking: {:?}", hacking.size());

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
    println!("Final minimization section");
    let mut steal1 = min_ph(&social_int, &steal_copy);
    steal1.reduce();

    println!("Final operation");
    let steal_exam = min_ph(&steal1, &hacking).spa();
    println!("steal_exam: {:?}", steal_exam.size());
    steal_exam.export_tra_with_labels("steal_exam_normal_f32")?;
    // steal_exam.reduce();
    // steal_exam.export_tra_with_labels("steal_exam_reduced")?;
    Ok(())
}

fn _stuxnet<F: PseudoField>() -> io::Result<()> {
    // -------------------------------------------------------------
    println!("Compromise Corporate Network section");
    let injection_via_usb = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-1", "1");

    let p2p_communication = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");
    let cnc_communication = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");
    let mut main_module_exec = min_ph(&p2p_communication, &cnc_communication);
    main_module_exec.reduce();

    let removable_media = Aph::<F, Bidiagonal<F>>::_new_erl(2, "1", "1");

    let network_shares = Aph::<F, Bidiagonal<F>>::_new_exp("-12", "1");
    let print_server_vuln = Aph::<F, Bidiagonal<F>>::_new_exp("-8", "1");
    let service_server_rpc_vuln = Aph::<F, Bidiagonal<F>>::_new_exp("-24", "1");
    let mut lan = min_phs(&vec![
        &network_shares,
        &print_server_vuln,
        &service_server_rpc_vuln,
    ])
    .unwrap();

    lan.reduce();

    let user_opens_win_cc_file_projects = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");

    let mut propagation = min_phs(&vec![
        &removable_media,
        &lan,
        &user_opens_win_cc_file_projects,
    ])
    .unwrap();

    propagation.reduce();

    let mut self_installation = con_ph(&main_module_exec, &propagation);
    self_installation.reduce();
    let mut compromise_corporate_network = max_ph(&injection_via_usb, &self_installation);
    compromise_corporate_network.reduce();

    // -------------------------------------------------------------
    println!("Attack industrial system section");

    let infection_of_control_pc = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-2", "10");
    let collect_data = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-1", "10");
    let plc_sends_false_data = Aph::<F, Bidiagonal<F>>::_new_exp("-30", "1");
    let intercept_in_out_signals = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-1", "10");
    let modify_out_signals = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");
    let cascade_centrifuges = Aph::<F, Bidiagonal<F>>::_new_erl(20, "-20", "1");

    let mut sys_300 = con_ph(&collect_data, &plc_sends_false_data);
    sys_300.reduce();
    let mut sys_400 = con_ph(&intercept_in_out_signals, &modify_out_signals);
    sys_400.reduce();
    let mut run_modified_code_on_plc = min_ph(&sys_300, &sys_400);
    run_modified_code_on_plc.reduce();
    let mut scada_system_compromised = con_ph(&infection_of_control_pc, &run_modified_code_on_plc);
    scada_system_compromised.reduce();
    let mut attack_industrial_system = con_ph(&scada_system_compromised, &cascade_centrifuges);
    attack_industrial_system.reduce();

    // -------------------------------------------------------------
    // println!("Final minimization section");
    let mut stuxnet = con_ph(&compromise_corporate_network, &attack_industrial_system);
    stuxnet.reduce();
    stuxnet.export_tra_with_labels("stuxnet")?;
    Ok(())
}
