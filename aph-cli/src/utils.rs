use aph::aph::new_erl;
use aph::aph::new_exp;
use aph::formats::ft::utils::ConstructionMethod;
use aph::formats::ft::utils::RoundMode;
use aph::linalg::fields::Round;
use aph::linalg::fields::float64::Float64;
use aph::linalg::fields::inari_int::Interval;
use aph::linalg::fields::rational::Rational;

use aph::{
    aph::Aph,
    linalg::{Vector, fields::*},
    operations::{convolution_bidiagonal, maximum_bidiagonal, minimum_bidiagonal},
    representation::bidiagonal::Bidiagonal,
};
use clap::{Parser, ValueEnum};
use log::info;
use serde::{Deserialize, Serialize};
use std::io;
use std::thread;

#[derive(Parser, Debug, Clone, Copy)]
pub struct Config {
    /// Reduce the APH using Reza Pulungans Algorithm.
    #[arg(long, default_value_t = false)]
    pub reduce: bool,
    /// Numeric Field to employ.
    #[arg(value_enum, short, long, default_value_t=NumericField::Rational)]
    pub numeric_field: NumericField,
    /// Round the APH [Round::Up] or, [Round::Down]. [Round::Nearest] does not round the distribution.
    #[arg(value_enum, short, long, default_value_t=Round::Nearest)]
    pub round: Round,
    /// For DFT only, First split tree in independent modules, and process from bottom up.
    #[arg(long, default_value_t = false)]
    pub modularise: bool,
    /// Use one of the known Models, they come from Reza Pulungan Thesis, or from the
    /// [FFORT benchmark collection](https://dftbenchmarks.utwente.nl/ffort/ffort.php) and are Fault Trees or Attack Trees.
    #[arg(value_enum, short, long, default_value_t=Model::Ex37, conflicts_with="input")]
    pub model: Model,
    /// Which method to use when constructing the Fault Tree.
    #[arg(value_enum, long, default_value_t=ConstructionMethod::Sparse, requires="input")]
    pub method: ConstructionMethod,
    /// Which method to use to compute bounds
    #[arg(value_enum, long, default_value_t=RoundMode::Mix, requires="round")]
    pub mode: RoundMode,
}

#[derive(Debug, Copy, Clone, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NumericField {
    Rational,
    F64,
    Float,
    Interval,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Model {
    Ex37,
    Ft3p2m,
    StealExam,
    Stuxnet,
    EMA,
    CITD,
    APPF,
    RBC,
}

#[allow(unused)]
pub fn run_models(config: Config) -> (usize, String, io::Result<(String)>) {
    info!(
        "Executing model '{:?}' with field {:?}.",
        config.model, config.numeric_field
    );
    match config.numeric_field {
        NumericField::Rational => {
            choose_model::<Rational>(config.numeric_field, config.model, config.round)
        }
        NumericField::Float | NumericField::F64 => {
            choose_model::<Float64>(config.numeric_field, config.model, config.round)
        }
        NumericField::Interval => {
            choose_model::<Interval>(config.numeric_field, config.model, config.round)
        }
    }
}

fn choose_model<F: PseudoField + 'static>(
    field: NumericField,
    model: Model,
    round: Round,
) -> (usize, String, io::Result<String>) {
    let mut suffix = format!(
        "{}-{}-{}",
        serde_json::to_string(&model).unwrap().replace("\"", ""),
        serde_json::to_string(&field).unwrap().replace("\"", ""),
        round
    );
    let ph = match model {
        Model::Ex37 => ex3_7::<F>(round),
        Model::CITD => citd::<F>(round),
        Model::APPF => appf::<F>(round),
        Model::StealExam => steal_exam::<F>(round),
        Model::Stuxnet => stuxnet::<F>(round),
        Model::Ft3p2m => {
            let phases = 30;
            suffix.push_str(&format!("_ph={}", phases));
            ft3p2m::<F>(phases, round)
        }
        Model::EMA => ema::<F>(round),
        Model::RBC => rbc::<F>(round),
    };

    let mttf = match ph.expected_value() {
        Ok(v) => v.to_string(),
        Err(_) => F::zero().to_string(),
    };
    (ph.size(), mttf, ph.ma_export(&format!("models/{}", suffix)))
}

/// From the Thesis work of Reza Pulungan.
pub fn ex3_7<F: PseudoField>(__round: Round) -> Aph<F, Bidiagonal<F>> {
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
    ph
}

pub fn citd<F: PseudoField>(round: Round) -> Aph<F, Bidiagonal<F>> {
    let find_lan_access_port = new_erl(5, -1.0);
    let spoof_mac_address = new_erl(2, -1.0);
    let find_wlan = new_erl(5, -5.0);
    let break_wpa_keys = new_erl(2, -2.0);

    let access_lan = maximum_bidiagonal(
        [&find_lan_access_port, &spoof_mac_address].into_iter(),
        round,
        RoundMode::Mix,
    );
    let access_wlan = maximum_bidiagonal(
        [&find_wlan, &break_wpa_keys].into_iter(),
        round,
        RoundMode::Mix,
    );

    let gain_access_to_private_networks = minimum_bidiagonal(
        [&access_lan, &access_wlan].into_iter(),
        round,
        RoundMode::Mix,
    );
    let get_credentials = new_erl(2, -10.0);

    let access_home_network = maximum_bidiagonal(
        [&get_credentials, &gain_access_to_private_networks].into_iter(),
        round,
        RoundMode::Mix,
    );

    let exploit_software_vulnerability = new_exp(-1.0);
    let run_malicous_script = new_erl(10, -1.0);

    convolution_bidiagonal(
        [
            access_home_network,
            exploit_software_vulnerability,
            run_malicous_script,
        ]
        .iter(),
        round,
    )
}

pub fn appf<F: PseudoField>(round: Round) -> Aph<F, Bidiagonal<F>> {
    let guessing = new_exp(-3.0 / 5.0);
    let dictionary = new_exp(-3.0 / 8.0);
    let bruteforce = new_exp(-7.0 / 5.0);
    let password_intercept = new_exp(-3.0 / 7.0);
    let email_trap_execution = new_exp(-9.0 / 6.0);
    let phone_trap_execution = new_exp(-2.0);
    let generic_reconnaissance = new_exp(-5.0 / 6.0);
    let payload_crafting = new_exp(-9.0 / 7.0);
    let email_file_execution = new_exp(-1.0 / 3.0);
    let physical_reconnaissance = new_exp(-1.0 / 5.0);
    let keylogger_local_installation = new_exp(-1.0);

    let physical = maximum_bidiagonal(
        [&physical_reconnaissance, &keylogger_local_installation].into_iter(),
        round,
        RoundMode::Mix,
    );

    let remote = maximum_bidiagonal(
        [
            &generic_reconnaissance,
            &payload_crafting,
            &email_file_execution,
        ]
        .into_iter(),
        round,
        RoundMode::Mix,
    );

    let key_logger_installation_alternatives =
        minimum_bidiagonal([&remote, &physical].into_iter(), round, RoundMode::Mix);

    let key_logger = maximum_bidiagonal(
        [&key_logger_installation_alternatives, &password_intercept].into_iter(),
        round,
        RoundMode::Mix,
    );

    let email_phone_trap = minimum_bidiagonal(
        [&email_trap_execution, &phone_trap_execution].into_iter(),
        round,
        RoundMode::Mix,
    );

    let social_engineering = maximum_bidiagonal(
        [&generic_reconnaissance, &email_phone_trap].into_iter(),
        round,
        RoundMode::Mix,
    );

    let password_attack = minimum_bidiagonal(
        [&social_engineering, &key_logger].into_iter(),
        round,
        RoundMode::Mix,
    );

    let cracking_alt = minimum_bidiagonal(
        [&guessing, &dictionary, &bruteforce].into_iter(),
        round,
        RoundMode::Mix,
    );

    minimum_bidiagonal(
        [&cracking_alt, &password_attack].into_iter(),
        round,
        RoundMode::Mix,
    )
}

pub fn steal_exam<F: PseudoField>(round: Round) -> Aph<F, Bidiagonal<F>> {
    let prep_bribe = new_erl(3, -4.0);
    let prep_threa = new_erl(2, -3.0);

    let prepare = minimum_bidiagonal(
        [&prep_bribe, &prep_threa].into_iter(),
        round,
        RoundMode::Mix,
    );

    let approach = new_exp(-4.0);
    let social_int = convolution_bidiagonal([&prepare, &approach].into_iter(), round);

    let hack = new_erl(6, -4.0);
    let keylogger = new_erl(4, -3.0);
    let guessing = new_exp(-0.5);
    let crack_firewall = new_erl(4, -3.0);
    let penetrate_repo = new_exp(-4.0);

    let acq_pass = minimum_bidiagonal([&keylogger, &guessing].into_iter(), round, RoundMode::Mix);

    let mailbox = minimum_bidiagonal([&hack, &acq_pass].into_iter(), round, RoundMode::Mix);
    let repo = convolution_bidiagonal([&crack_firewall, &penetrate_repo].into_iter(), round);

    let hacking = minimum_bidiagonal([&mailbox, &repo].into_iter(), round, RoundMode::Mix);

    let loc_office = new_exp(-10.0);
    let steal_key = new_erl(4, -6.0);

    let break_into_ofi = new_erl(4, -2.0);
    let find_prints = new_exp(-24.0);

    let get_access = minimum_bidiagonal(
        [&steal_key, &break_into_ofi].into_iter(),
        round,
        RoundMode::Mix,
    );

    let steal_copy1 = convolution_bidiagonal([&loc_office, &get_access].into_iter(), round);
    let steal_copy = convolution_bidiagonal([&steal_copy1, &find_prints].into_iter(), round);

    let steal1 = minimum_bidiagonal(
        [&social_int, &steal_copy].into_iter(),
        round,
        RoundMode::Mix,
    );

    minimum_bidiagonal([&steal1, &hacking].into_iter(), round, RoundMode::Mix)
}

pub fn stuxnet<F: PseudoField>(round: Round) -> Aph<F, Bidiagonal<F>> {
    let injection_via_usb = new_erl(2, -1.0);
    let p2p_communication = new_exp(-1.0);
    let cnc_communication = new_exp(-1.0);
    let main_module_exec = minimum_bidiagonal(
        [&p2p_communication, &cnc_communication].into_iter(),
        round,
        RoundMode::Mix,
    );

    let removable_media = new_erl(2, -1.0);

    let network_shares = new_exp(-12.0);
    let print_server_vuln = new_exp(-8.0);
    let service_server_rpc_vuln = new_exp(-24.0);
    let lan = minimum_bidiagonal(
        [
            &network_shares,
            &print_server_vuln,
            &service_server_rpc_vuln,
        ]
        .into_iter(),
        round,
        RoundMode::Mix,
    );

    let user_opens_win_cc_file_projects = new_exp(-1.0);

    let propagation = minimum_bidiagonal(
        [&removable_media, &lan, &user_opens_win_cc_file_projects].into_iter(),
        round,
        RoundMode::Mix,
    );

    let self_installation =
        convolution_bidiagonal([&main_module_exec, &propagation].into_iter(), round);

    let compromise_corporate_network = maximum_bidiagonal(
        [&injection_via_usb, &self_installation].into_iter(),
        round,
        RoundMode::Mix,
    );

    let infection_of_control_pc = new_erl(3, -2.0 / 10.0);
    let collect_data = new_erl(3, -1.0 / 10.0);
    let plc_sends_false_data = new_exp(-30.0);
    let intercept_in_out_signals = new_erl(3, -1.0 / 10.0);
    let modify_out_signals = new_exp(-1.0);
    let cascade_centrifuges = new_erl(20, -20.0);

    let sys_300 = convolution_bidiagonal([&collect_data, &plc_sends_false_data].into_iter(), round);

    let sys_400 = convolution_bidiagonal(
        [&intercept_in_out_signals, &modify_out_signals].into_iter(),
        round,
    );

    let run_modified_code_on_plc =
        minimum_bidiagonal([&sys_300, &sys_400].into_iter(), round, RoundMode::Mix);

    let scada_system_compromised = convolution_bidiagonal(
        [&infection_of_control_pc, &run_modified_code_on_plc].into_iter(),
        round,
    );

    let attack_industrial_system = convolution_bidiagonal(
        [&scada_system_compromised, &cascade_centrifuges].into_iter(),
        round,
    );

    convolution_bidiagonal(
        [&compromise_corporate_network, &attack_industrial_system].into_iter(),
        round,
    )
}

pub fn ft3p2m<F: PseudoField + 'static>(phases: usize, round: Round) -> Aph<F, Bidiagonal<F>> {
    let p: Aph<F, Bidiagonal<F>> = new_erl(phases, -(phases as f64) / 5.0);
    let m: Aph<F, Bidiagonal<F>> = new_erl(phases, -(phases as f64) / 3.0);
    let bus: Aph<F, Bidiagonal<F>> = new_erl(phases, -(phases as f64) / 7.0);

    let p_hand =
        thread::spawn(move || maximum_bidiagonal([&p, &p, &p].into_iter(), round, RoundMode::Mix));

    let m_hand =
        thread::spawn(move || maximum_bidiagonal([&m, &m].into_iter(), round, RoundMode::Mix));

    let processors = p_hand.join().unwrap();
    let memories = m_hand.join().unwrap();

    minimum_bidiagonal(
        [&processors, &memories, &bus].into_iter(),
        round,
        RoundMode::Mix,
    )
}

pub fn ema<F: PseudoField>(round: Round) -> Aph<F, Bidiagonal<F>> {
    let x1 = new_exp(-0.42e-6);
    let x2 = new_exp(-0.42e-6);
    let x3 = new_exp(-0.42e-6);
    let x4 = new_exp(-0.42e-6);
    let x5 = new_exp(-0.33e-6);
    let x6 = new_exp(-0.24e-6);
    let x7 = new_exp(-0.12e-6);
    let x8 = new_exp(-0.43e-6);
    let x9 = new_exp(-0.28e-6);
    let x10 = new_exp(-0.59e-6);
    let x11 = new_exp(-0.2e-6);
    let x12 = new_exp(-0.061e-6);
    let x13 = new_exp(-0.026e-6);
    let x14 = new_exp(-0.061e-6);
    let x15 = new_exp(-0.026e-6);
    let x16 = new_exp(-0.33e-6);
    let x17 = new_exp(-0.54e-6);
    let x18 = new_exp(-0.46e-6);
    let x19 = new_exp(-0.32e-6);
    let x20 = new_exp(-0.38e-6);

    // "CCF" fdep "X20" "X8" "X16";
    let x8 = minimum_bidiagonal([&x20, &x8].into_iter(), round, RoundMode::Mix);
    let x16 = minimum_bidiagonal([&x20, &x16].into_iter(), round, RoundMode::Mix);

    let m11b = minimum_bidiagonal([&x3, &x4].into_iter(), round, RoundMode::Mix);
    let m11a = minimum_bidiagonal([&x1, &x2].into_iter(), round, RoundMode::Mix);
    let m41 = convolution_bidiagonal([&x17, &x18, &x19].into_iter(), round);
    let m32 = minimum_bidiagonal([&x14, &x15].into_iter(), round, RoundMode::Mix);
    let m31 = minimum_bidiagonal([&x12, &x13].into_iter(), round, RoundMode::Mix);
    let m13a = minimum_bidiagonal([&x7, &x8].into_iter(), round, RoundMode::Mix);
    // "M13" fdep "M13A" "X9";
    let _x9 = minimum_bidiagonal([&x9, &m13a].into_iter(), round, RoundMode::Mix);

    let m12 = minimum_bidiagonal([&x5, &x6].into_iter(), round, RoundMode::Mix);
    let m11 = convolution_bidiagonal([&m11a, &m11b].into_iter(), round); //original: csp
    let m4 = minimum_bidiagonal([&x16, &m41].into_iter(), round, RoundMode::Mix);
    let m3 = convolution_bidiagonal([&m31, &m32].into_iter(), round); //original: csp
    let m2 = maximum_bidiagonal([&x10, &x11].into_iter(), round, RoundMode::Mix);
    let m1 = minimum_bidiagonal([&m11, &m12].into_iter(), round, RoundMode::Mix);

    minimum_bidiagonal([&m1, &m2, &m3, &m4].into_iter(), round, RoundMode::Mix)
}

pub fn rbc<F: PseudoField>(round: Round) -> Aph<F, Bidiagonal<F>> {
    let bus1 = new_exp(-4.4444e-6);
    let bus2 = new_exp(-4.4444e-6);
    let fpga1 = new_exp(-3.003e-9);
    let fpga2 = new_exp(-3.003e-9);
    let power_supply1 = new_exp(-1.8182e-5);
    let power_supply2 = new_exp(-1.8182e-5);
    let power_supply3 = new_exp(-1.8182e-5);
    let wancard1 = new_exp(-2.5e-6);
    let wancard2 = new_exp(-2.5e-6);
    let gsmrcard1 = new_exp(-5.7078e-6);
    let gsmrcard2 = new_exp(-5.7078e-6);
    let cpuboard1 = new_exp(-7.4074e-6);
    let cpuboard2 = new_exp(-7.4074e-6);
    let cpuboard3 = new_exp(-7.4074e-6);

    let voter = maximum_bidiagonal([&fpga1, &fpga2].into_iter(), round, RoundMode::Mix);
    let vot1 = maximum_bidiagonal([&cpuboard1, &cpuboard2].into_iter(), round, RoundMode::Mix);
    let vot2 = maximum_bidiagonal([&cpuboard2, &cpuboard3].into_iter(), round, RoundMode::Mix);
    let vot3 = maximum_bidiagonal([&cpuboard1, &cpuboard3].into_iter(), round, RoundMode::Mix);
    let cpucore = minimum_bidiagonal([&vot1, &vot2, &vot3].into_iter(), round, RoundMode::Mix);

    let tmr = minimum_bidiagonal([&cpucore, &voter].into_iter(), round, RoundMode::Mix);
    let gsmrinterface =
        maximum_bidiagonal([&gsmrcard1, &gsmrcard2].into_iter(), round, RoundMode::Mix);
    let system_bus = maximum_bidiagonal([&bus1, &bus2].into_iter(), round, RoundMode::Mix);
    let waninterface =
        maximum_bidiagonal([&wancard1, &wancard2].into_iter(), round, RoundMode::Mix);
    let power = maximum_bidiagonal(
        [&power_supply1, &power_supply2, &power_supply3].into_iter(),
        round,
        RoundMode::Mix,
    );

    minimum_bidiagonal(
        [&power, &waninterface, &system_bus, &gsmrinterface, &tmr].into_iter(),
        round,
        RoundMode::Mix,
    )
}
