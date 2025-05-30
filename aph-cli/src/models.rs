use std::fmt::Display;
use std::io;

use aph::{
    aph::{Aph, BidiagonalAph},
    linalg::{Vector, fields::*},
    operations::*,
    representation::Bidiagonal,
};
use clap::ValueEnum;
use log::warn;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Model {
    Ex37,
    Hacking,

    CITD,
    APPF,
    StealExam,
    Stuxnet,
    Ft3p2m,
}

impl Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Ex37 => Ok(write!(f, "ex37")?),
            Model::Hacking => Ok(write!(f, "hacking")?),
            Model::CITD => Ok(write!(f, "citd")?),
            Model::APPF => Ok(write!(f, "appf")?),
            Model::StealExam => Ok(write!(f, "steal-exam")?),
            Model::Stuxnet => Ok(write!(f, "stuxnet")?),
            Model::Ft3p2m => Ok(write!(f, "ft3p2m")?),
        }
    }
}

pub fn choose_model<F: PseudoField>(model: Model, suffix: &str) -> (usize, usize, io::Result<()>) {
    let (reductions, ph) = match model {
        Model::Ex37 => ex3_7::<F>(),
        Model::Hacking => hacking::<F>(),
        Model::CITD => citd::<F>(),
        Model::APPF => appf::<F>(),
        Model::StealExam => steal_exam::<F>(),
        Model::Stuxnet => stuxnet::<F>(),
        Model::Ft3p2m => ft3p2m::<F>(5),
    };

    let _ = ph.export_to_ma(&format!("models/{}_{}", model, suffix));
    match ph
        .to_coxian()
        .ctmc_export(&format!("models/{}_{}", model, suffix))
    {
        Ok(()) => (reductions, ph.size(), Ok(())),
        Err(e) => {
            warn!("Something went wrong when exporting the aph: {}", e);
            (reductions, ph.size(), Err(e))
        }
    }
}

pub fn ex3_7<F: PseudoField>() -> (usize, Aph<F, Bidiagonal<F>>) {
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
    let total_red = ph.reduce();
    (total_red, ph)
}

pub fn hacking<F: PseudoField>() -> (usize, Aph<F, Bidiagonal<F>>) {
    let new_exp = Aph::<F, Bidiagonal<_>>::new_exp;
    let new_erl = Aph::<F, Bidiagonal<_>>::new_erl;
    let mut total_red = 0;

    let hack = new_erl(26, "-4", "1");
    let keylogger = new_erl(14, "-3", "1");
    let crack_firewall = new_erl(17, "-3", "1");

    let guessing = new_exp("-1", "2");
    let penetrate_repo = new_exp("-4", "1");

    let mut acq_pass = min_minimal(&keylogger, &guessing);
    total_red += acq_pass.reduce();

    let mut mailbox = min_minimal(&hack, &acq_pass);
    total_red += mailbox.reduce();
    let mut repo = con_ph(&crack_firewall, &penetrate_repo);
    total_red += repo.reduce();

    let mut hacking = min_minimal(&mailbox, &repo);
    total_red += hacking.reduce();

    (total_red, hacking)
}

pub fn citd<F: PseudoField>() -> (usize, Aph<F, Bidiagonal<F>>) {
    let new_exp = Aph::<F, Bidiagonal<_>>::new_exp;
    let new_erl = Aph::<F, Bidiagonal<_>>::new_erl;
    let mut total_red = 0;

    let find_lan_access_port = new_erl(5, "-1", "1");
    let spoof_mac_address = new_erl(2, "-1", "2");
    let find_wlan = new_erl(5, "-5", "1");
    let break_wpa_keys = new_erl(2, "-2", "1");

    let mut access_lan = max_minimal(&find_lan_access_port, &spoof_mac_address);
    total_red += access_lan.reduce();
    let mut access_wlan = max_minimal(&find_wlan, &break_wpa_keys);
    total_red += access_wlan.reduce();

    let mut gain_access_to_private_networks = min_minimal(&access_lan, &access_wlan);
    total_red += gain_access_to_private_networks.reduce();
    let get_credentials = new_erl(2, "-10", "1");

    let mut access_home_network = max_minimal(&get_credentials, &gain_access_to_private_networks);
    total_red += access_home_network.reduce();

    let exploit_software_vulnerability = new_exp("-1", "1");
    let run_malicous_script = new_erl(10, "-1", "2");

    let mut compromise_iot_device = apply_many(
        [
            access_home_network,
            exploit_software_vulnerability,
            run_malicous_script,
        ]
        .iter(),
        con_ph,
    )
    .unwrap();

    total_red += compromise_iot_device.reduce();

    (total_red, compromise_iot_device)
}

pub fn appf<F: PseudoField>() -> (usize, Aph<F, Bidiagonal<F>>) {
    let new_exp = Aph::<F, Bidiagonal<_>>::new_exp;
    let _new_erl = Aph::<F, Bidiagonal<_>>::new_erl;
    let mut total_red = 0;

    let guessing = new_exp("-3", "5");
    let dictionary = new_exp("-3", "8");
    let bruteforce = new_exp("-7", "5");
    let password_intercept = new_exp("-3", "7");
    let email_trap_execution = new_exp("-9", "6");
    let phone_trap_execution = new_exp("-2", "1");
    let generic_reconnaissance = new_exp("-5", "6");
    let payload_crafting = new_exp("-9", "7");
    let email_file_execution = new_exp("-1", "3");
    let physical_reconnaissance = new_exp("-1", "5");
    let keylogger_local_installation = new_exp("-1", "1");

    let mut physical = max_minimal(&physical_reconnaissance, &keylogger_local_installation);
    total_red += physical.reduce();

    let mut remote1 = max_minimal(&generic_reconnaissance, &payload_crafting);
    total_red += remote1.reduce();

    let mut remote = max_minimal(&remote1, &email_file_execution);
    total_red += remote.reduce();

    let mut key_logger_installation_alternatives = min_minimal(&remote, &physical);
    total_red += key_logger_installation_alternatives.reduce();

    let mut key_logger = max_minimal(&key_logger_installation_alternatives, &password_intercept);
    total_red += key_logger.reduce();

    let mut email_phone_trap = min_minimal(&email_trap_execution, &phone_trap_execution);

    total_red += email_phone_trap.reduce();

    let mut social_engineering = max_minimal(&generic_reconnaissance, &email_phone_trap);
    total_red += social_engineering.reduce();

    // Almost all reductions happen here because the sums of the rates are not disjunct
    let mut password_attack = min_minimal(&social_engineering, &key_logger);

    total_red += password_attack.reduce();

    let mut cracking_alt1 = min_minimal(&guessing, &dictionary);

    total_red += cracking_alt1.reduce();

    let mut cracking_alt = min_minimal(&cracking_alt1, &bruteforce);

    total_red += cracking_alt.reduce();

    let mut password_attack_success = min_minimal(&cracking_alt, &password_attack);

    total_red += password_attack_success.reduce();

    (total_red, password_attack_success)
}

pub fn steal_exam<F: PseudoField>() -> (usize, Aph<F, Bidiagonal<F>>) {
    let new_exp = Aph::<F, Bidiagonal<_>>::new_exp;
    let new_erl = Aph::<F, Bidiagonal<_>>::new_erl;
    let mut total_red = 0;

    let prep_bribe = new_erl(3, "-4", "1");

    let prep_threa = new_erl(2, "-3", "1");
    let prepare = min_minimal(&prep_bribe, &prep_threa);

    let approach = new_exp("-4", "1");
    let mut social_int = con_ph(&prepare, &approach);

    total_red += social_int.reduce();

    let hack = new_erl(6, "-4", "1");
    let keylogger = new_erl(4, "-3", "1");
    let guessing = new_exp("-1", "2");
    let crack_firewall = new_erl(4, "-3", "1");
    let penetrate_repo = new_exp("-4", "1");

    let mut acq_pass = min_minimal(&keylogger, &guessing);
    total_red += acq_pass.reduce();

    let mut mailbox = min_minimal(&hack, &acq_pass);
    let mut repo = con_ph(&crack_firewall, &penetrate_repo);

    total_red += mailbox.reduce();
    total_red += repo.reduce();

    let mut hacking = min_minimal(&mailbox, &repo);
    total_red += hacking.reduce();

    let loc_office = new_exp("-10", "1");
    let steal_key = new_erl(4, "-6", "1");

    let break_into_ofi = new_erl(4, "-2", "1");
    let find_prints = new_exp("-24", "1");

    let mut get_access = min_minimal(&steal_key, &break_into_ofi);
    total_red += get_access.reduce();

    let steal_copy1 = con_ph(&loc_office, &get_access);
    let mut steal_copy = con_ph(&steal_copy1, &find_prints);

    total_red += steal_copy.reduce();
    let mut steal1 = min_minimal(&social_int, &steal_copy);
    total_red += steal1.reduce();

    let mut steal_exam = min_minimal(&steal1, &hacking);
    total_red += steal_exam.reduce();

    (total_red, steal_exam)
}

pub fn stuxnet<F: PseudoField>() -> (usize, Aph<F, Bidiagonal<F>>) {
    let new_exp = Aph::<F, Bidiagonal<_>>::new_exp;
    let new_erl = Aph::<F, Bidiagonal<_>>::new_erl;
    let mut total_red = 0;

    let injection_via_usb = new_erl(2, "-1", "1");
    let p2p_communication = new_exp("-1", "1");
    let cnc_communication = new_exp("-1", "1");
    let mut main_module_exec = min_minimal(&p2p_communication, &cnc_communication);

    total_red += main_module_exec.reduce();

    let removable_media = new_erl(2, "1", "1");

    let network_shares = new_exp("-12", "1");
    let print_server_vuln = new_exp("-8", "1");
    let service_server_rpc_vuln = new_exp("-24", "1");
    let mut lan = apply_many(
        [network_shares, print_server_vuln, service_server_rpc_vuln].iter(),
        min_minimal,
    )
    .unwrap();

    total_red += lan.reduce();

    let user_opens_win_cc_file_projects = new_exp("-1", "1");

    let mut propagation = apply_many(
        [removable_media, lan, user_opens_win_cc_file_projects].iter(),
        min_minimal,
    )
    .unwrap();

    total_red += propagation.reduce();

    let mut self_installation = con_ph(&main_module_exec, &propagation);

    total_red += self_installation.reduce();

    let mut compromise_corporate_network = max_minimal(&injection_via_usb, &self_installation);

    total_red += compromise_corporate_network.reduce();

    let infection_of_control_pc = new_erl(3, "-2", "10");
    let collect_data = new_erl(3, "-1", "10");
    let plc_sends_false_data = new_exp("-30", "1");
    let intercept_in_out_signals = new_erl(3, "-1", "10");
    let modify_out_signals = new_exp("-1", "1");
    let cascade_centrifuges = new_erl(20, "-20", "1");

    let mut sys_300 = con_ph(&collect_data, &plc_sends_false_data);

    total_red += sys_300.reduce();

    let mut sys_400 = con_ph(&intercept_in_out_signals, &modify_out_signals);

    total_red += sys_400.reduce();

    let mut run_modified_code_on_plc = min_minimal(&sys_300, &sys_400);

    total_red += run_modified_code_on_plc.reduce();

    let mut scada_system_compromised = con_ph(&infection_of_control_pc, &run_modified_code_on_plc);

    total_red += scada_system_compromised.reduce();

    let mut attack_industrial_system = con_ph(&scada_system_compromised, &cascade_centrifuges);

    total_red += attack_industrial_system.reduce();

    let mut stuxnet = con_ph(&compromise_corporate_network, &attack_industrial_system);

    total_red += stuxnet.reduce();

    (total_red, stuxnet)
}

pub fn ft3p2m<F: PseudoField>(phases: usize) -> (usize, Aph<F, Bidiagonal<F>>) {
    let new_erl = BidiagonalAph::<F>::new_erl;
    let mut total_red = 0;
    let p = new_erl(phases, &format!("-{}", phases), "5");
    let m = new_erl(phases, &format!("-{}", phases), "3");
    let bus = new_erl(phases, &format!("-{}", phases), "7");

    // let p = con_ph(
    //     &new_erl(phases / 2, &format!("-{}", phases / 2), "5"),
    //     &new_erl(phases / 2, &format!("-{}", phases / 2), "15"),
    // );
    // let m = con_ph(
    //     &new_erl(phases / 2, &format!("-{}", phases / 2), "3"),
    //     &new_erl(phases / 2, &format!("-{}", phases / 2), "13"),
    // );
    // let bus = con_ph(
    //     &new_erl(phases / 2, &format!("-{}", phases / 2), "7"),
    //     &new_erl(phases / 2, &format!("-{}", phases / 2), "17"),
    // );

    let mut processors = apply_many([&p, &p, &p].into_iter(), max_minimal).unwrap();
    total_red += processors.reduce();
    let mut memories = apply_many([&m, &m].into_iter(), max_minimal).unwrap();
    total_red += memories.reduce();
    let mut toplevel = apply_many([&processors, &memories, &bus].into_iter(), min_minimal).unwrap();
    total_red += toplevel.reduce();

    (total_red, toplevel)
}
