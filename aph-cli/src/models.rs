use std::io;

use aph::{
    linalg::{fields::PseudoField, Vector},
    operations::*,
    representation::Bidiagonal,
    Aph,
};
use clap::ValueEnum;
use log::info;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Model {
    Ex37,
    Hacking,
    HackingSmall,
    CITD,
    APPF,
    StealExam,
    Stuxnet,
}

pub fn choose_model<F: PseudoField>(model: Model) -> io::Result<()> {
    match model {
        Model::Ex37 => _ex3_7::<F>(),
        Model::Hacking => _hacking::<F>(),
        Model::HackingSmall => _hacking_small::<F>(),
        Model::CITD => _citd::<F>(),
        Model::APPF => _appf::<F>(),
        Model::StealExam => _steal_exam::<F>(),
        Model::Stuxnet => _stuxnet::<F>(),
    }
}

pub fn _ex3_7<F: PseudoField>() -> io::Result<()> {
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
    info!("Number of Reductions: {:?}", total_red);
    ph.to_coxian().export("ex3_7")?;
    Ok(())
}

pub fn _hacking_small<F: PseudoField>() -> io::Result<()> {
    let mut total_red = 0;
    let hack = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-4", "1");
    let keylogger = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-3", "1");
    let guessing = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "2");
    let crack_firewall = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-3", "1");
    let penetrate_repo = Aph::<F, Bidiagonal<F>>::_new_exp("-4", "1");

    let mut acq_pass = min_ph(&keylogger, &guessing);
    total_red += acq_pass.reduce();

    let mut mailbox = min_ph(&hack, &acq_pass);
    total_red += mailbox.reduce();
    let mut repo = con_ph(&crack_firewall, &penetrate_repo);
    total_red += repo.reduce();

    let mut hacking = min_ph(&mailbox, &repo);

    total_red += hacking.reduce();

    info!("Number of Reductions: {:?}", total_red);
    hacking.to_coxian().export("hacking_small")?;
    Ok(())
}

pub fn _citd<F: PseudoField>() -> io::Result<()> {
    let mut total_red = 0;
    let find_lan_access_port = Aph::<F, Bidiagonal<F>>::_new_erl(5, "-1", "1");
    let spoof_mac_address = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-1", "2");
    let find_wlan = Aph::<F, Bidiagonal<F>>::_new_erl(5, "-5", "1");
    let break_wpa_keys = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-2", "1");

    let mut access_lan = max_ph(&find_lan_access_port, &spoof_mac_address);
    total_red += access_lan.reduce();
    let mut access_wlan = max_ph(&find_wlan, &break_wpa_keys);
    total_red += access_wlan.reduce();

    let mut gain_access_to_private_networks = min_ph(&access_lan, &access_wlan);
    total_red += gain_access_to_private_networks.reduce();
    let get_credentials = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-10", "1");

    let mut access_home_network = max_ph(&get_credentials, &gain_access_to_private_networks);
    total_red += access_home_network.reduce();

    let exploit_software_vulnerability = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");
    let run_malicous_script = Aph::<F, Bidiagonal<F>>::_new_erl(10, "-1", "2");

    let mut compromise_iot_device = con_phs(&vec![
        &access_home_network,
        &exploit_software_vulnerability,
        &run_malicous_script,
    ])
    .unwrap();

    total_red += compromise_iot_device.reduce();

    info!("Number of Reductions: {:?}", total_red);
    compromise_iot_device.to_coxian().export("citd")?;
    // compromise_iot_device._export_to_tra("citd_bidi_red_)?;
    Ok(())
}

pub fn _hacking<F: PseudoField>() -> io::Result<()> {
    let mut total_red = 0;
    let hack = Aph::<F, Bidiagonal<F>>::_new_erl(6, "-4", "1");
    let keylogger = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    let guessing = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "2");
    let crack_firewall = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    let penetrate_repo = Aph::<F, Bidiagonal<F>>::_new_exp("-4", "1");

    let mut acq_pass = min_ph(&keylogger, &guessing);
    total_red += acq_pass.reduce();

    let mut mailbox = min_ph(&hack, &acq_pass);
    total_red += mailbox.reduce();
    let mut repo = con_ph(&crack_firewall, &penetrate_repo);
    total_red += repo.reduce();

    let mut hacking = min_ph(&mailbox, &repo);
    total_red += hacking.reduce();

    info!("Number of Reductions: {:?}", total_red);
    hacking.to_coxian().export("hacking")?;
    Ok(())
}

pub fn _appf<F: PseudoField>() -> io::Result<()> {
    let mut total_red = 0;
    let guessing = Aph::<F, Bidiagonal<F>>::_new_exp("-3", "5");
    let dictionary = Aph::<F, Bidiagonal<F>>::_new_exp("-3", "8");
    let bruteforce = Aph::<F, Bidiagonal<F>>::_new_exp("-7", "5");
    let password_intercept = Aph::<F, Bidiagonal<F>>::_new_exp("-3", "7");
    let email_trap_execution = Aph::<F, Bidiagonal<F>>::_new_exp("-9", "6");
    let phone_trap_execution = Aph::<F, Bidiagonal<F>>::_new_exp("-2", "1");
    let generic_reconnaissance = Aph::<F, Bidiagonal<F>>::_new_exp("-5", "6");
    let payload_crafting = Aph::<F, Bidiagonal<F>>::_new_exp("-9", "7");
    let email_file_execution = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "3");
    let physical_reconnaissance = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "5");
    let keylogger_local_installation = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");

    let mut physical = max_ph(&physical_reconnaissance, &keylogger_local_installation);
    total_red += physical.reduce();

    let mut remote = max_phs(&vec![
        &generic_reconnaissance,
        &payload_crafting,
        &email_file_execution,
    ])
    .unwrap();
    total_red += remote.reduce();

    let mut key_logger_installation_alternatives = min_ph(&remote, &physical);
    total_red += key_logger_installation_alternatives.reduce();

    let mut key_logger = max_ph(&key_logger_installation_alternatives, &password_intercept);
    total_red += key_logger.reduce();

    let mut email_phone_trap_execution_and_user_trapped =
        min_ph(&email_trap_execution, &phone_trap_execution);
    total_red += email_phone_trap_execution_and_user_trapped.reduce();

    let mut social_engineering = max_ph(
        &generic_reconnaissance,
        &email_phone_trap_execution_and_user_trapped,
    );
    total_red += social_engineering.reduce();

    let mut password_attack = min_ph(&social_engineering, &key_logger);
    total_red += password_attack.reduce();

    let mut cracking_alternatives = min_phs(&vec![&guessing, &dictionary, &bruteforce]).unwrap();
    total_red += cracking_alternatives.reduce();

    let mut password_attack_success = min_ph(&cracking_alternatives, &password_attack);
    total_red += password_attack_success.reduce();

    info!("Number of Reductions: {:?}", total_red);
    password_attack_success.to_coxian().export("appf")?;
    Ok(())
}

pub fn _steal_exam<F: PseudoField>() -> io::Result<()> {
    let mut total_red = 0;
    let prep_bribe = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-4", "1");

    let prep_threa = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-3", "1");
    let prepare = min_ph(&prep_bribe, &prep_threa);

    let approach = Aph::<F, Bidiagonal<F>>::_new_exp("-4", "1");
    let mut social_int = con_ph(&prepare, &approach);

    total_red += social_int.reduce();

    let hack = Aph::<F, Bidiagonal<F>>::_new_erl(6, "-4", "1");
    let keylogger = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    let guessing = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "2");
    let crack_firewall = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
    let penetrate_repo = Aph::<F, Bidiagonal<F>>::_new_exp("-4", "1");

    let mut acq_pass = min_ph(&keylogger, &guessing);
    total_red += acq_pass.reduce();

    let mut mailbox = min_ph(&hack, &acq_pass);
    let mut repo = con_ph(&crack_firewall, &penetrate_repo);

    total_red += mailbox.reduce();
    total_red += repo.reduce();

    let mut hacking = min_ph(&mailbox, &repo);
    total_red += hacking.reduce();

    let loc_office = Aph::<F, Bidiagonal<F>>::_new_exp("-10", "1");
    let steal_key = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-6", "1");

    let break_into_ofi = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-2", "1");
    let find_prints = Aph::<F, Bidiagonal<F>>::_new_exp("-24", "1");

    let mut get_access = min_ph(&steal_key, &break_into_ofi);
    total_red += get_access.reduce();

    let steal_copy1 = con_ph(&loc_office, &get_access);
    let mut steal_copy = con_ph(&steal_copy1, &find_prints);

    total_red += steal_copy.reduce();
    let mut steal1 = min_ph(&social_int, &steal_copy);
    total_red += steal1.reduce();

    let mut steal_exam = min_ph(&steal1, &hacking);
    total_red += steal_exam.reduce();

    info!("Number of Reductions: {:?}", total_red);
    steal_exam.to_coxian().export("steal_exam")?;
    Ok(())
}

pub fn _stuxnet<F: PseudoField>() -> io::Result<()> {
    let mut total_red = 0;
    let injection_via_usb = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-1", "1");

    let p2p_communication = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");
    let cnc_communication = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");
    let mut main_module_exec = min_ph(&p2p_communication, &cnc_communication);
    total_red += main_module_exec.reduce();

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

    total_red += lan.reduce();

    let user_opens_win_cc_file_projects = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");

    let mut propagation = min_phs(&vec![
        &removable_media,
        &lan,
        &user_opens_win_cc_file_projects,
    ])
    .unwrap();

    total_red += propagation.reduce();

    let mut self_installation = con_ph(&main_module_exec, &propagation);
    total_red += self_installation.reduce();
    let mut compromise_corporate_network = max_ph(&injection_via_usb, &self_installation);
    total_red += compromise_corporate_network.reduce();

    let infection_of_control_pc = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-2", "10");
    let collect_data = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-1", "10");
    let plc_sends_false_data = Aph::<F, Bidiagonal<F>>::_new_exp("-30", "1");
    let intercept_in_out_signals = Aph::<F, Bidiagonal<F>>::_new_erl(3, "-1", "10");
    let modify_out_signals = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "1");
    let cascade_centrifuges = Aph::<F, Bidiagonal<F>>::_new_erl(20, "-20", "1");

    let mut sys_300 = con_ph(&collect_data, &plc_sends_false_data);
    total_red += sys_300.reduce();
    let mut sys_400 = con_ph(&intercept_in_out_signals, &modify_out_signals);
    total_red += sys_400.reduce();
    let mut run_modified_code_on_plc = min_ph(&sys_300, &sys_400);
    total_red += run_modified_code_on_plc.reduce();
    let mut scada_system_compromised = con_ph(&infection_of_control_pc, &run_modified_code_on_plc);
    total_red += scada_system_compromised.reduce();
    let mut attack_industrial_system = con_ph(&scada_system_compromised, &cascade_centrifuges);
    total_red += attack_industrial_system.reduce();

    let mut stuxnet = con_ph(&compromise_corporate_network, &attack_industrial_system);
    total_red += stuxnet.reduce();
    info!("Number of Reductions: {:?}", total_red);
    stuxnet.to_coxian().export("stuxnet")?;
    Ok(())
}

// pub fn _vot3of6<F: PseudoField>() -> io::Result<()> {
//     let _b1 = Aph::<F, Bidiagonal<F>>::_new_exp("-5", "10");
//     let _b2 = Aph::<F, Bidiagonal<F>>::_new_exp("-77", "10");
//     let _b3 = Aph::<F, Bidiagonal<F>>::_new_erl(4, "-3", "1");
//     let _b4 = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-4", "10");
//     let _b5 = Aph::<F, Bidiagonal<F>>::_new_erl(2, "-1", "20");
//     let _b6 = Aph::<F, Bidiagonal<F>>::_new_exp("-1", "10");
//     // let mut b1_b2_b3 = max_phs(&vec![&_b1, &_b2, &_b3]).unwrap();
//     b1_b2_b3.reduce();
//     // let mut b1_b2_b4 = max_phs(&vec![&_b1, &_b2, &_b4]).unwrap();
//     b1_b2_b4.reduce();
//     // let mut b1_b2_b5 = max_phs(&vec![&_b1, &_b2, &_b5]).unwrap();
//     b1_b2_b5.reduce();
//     // let mut b1_b2_b6 = max_phs(&vec![&_b1, &_b2, &_b6]).unwrap();
//     b1_b2_b6.reduce();
//     // let mut b1_b3_b4 = max_phs(&vec![&_b1, &_b3, &_b4]).unwrap();
//     b1_b3_b4.reduce();
//     // let mut b1_b3_b5 = max_phs(&vec![&_b1, &_b3, &_b5]).unwrap();
//     b1_b3_b5.reduce();
//     // let mut b1_b3_b6 = max_phs(&vec![&_b1, &_b3, &_b6]).unwrap();
//     b1_b3_b6.reduce();
//     // let mut b1_b4_b5 = max_phs(&vec![&_b1, &_b4, &_b5]).unwrap();
//     b1_b4_b5.reduce();
//     // let mut b1_b4_b6 = max_phs(&vec![&_b1, &_b4, &_b6]).unwrap();
//     b1_b4_b6.reduce();
//     // let mut b1_b5_b6 = max_phs(&vec![&_b1, &_b5, &_b6]).unwrap();
//     b1_b5_b6.reduce();
//     // let mut b2_b3_b4 = max_phs(&vec![&_b2, &_b3, &_b4]).unwrap();
//     b2_b3_b4.reduce();
//     // let mut b2_b3_b5 = max_phs(&vec![&_b2, &_b3, &_b5]).unwrap();
//     b2_b3_b5.reduce();
//     // let mut b2_b3_b6 = max_phs(&vec![&_b2, &_b3, &_b6]).unwrap();
//     b2_b3_b6.reduce();
//     // let mut b2_b4_b5 = max_phs(&vec![&_b2, &_b4, &_b5]).unwrap();
//     b2_b4_b5.reduce();
//     // let mut b2_b4_b6 = max_phs(&vec![&_b2, &_b4, &_b6]).unwrap();
//     b2_b4_b6.reduce();
//     // let mut b2_b5_b6 = max_phs(&vec![&_b2, &_b5, &_b6]).unwrap();
//     b2_b5_b6.reduce();
//     let mut b3_b4_b5 = max_phs(&vec![&_b3, &_b4, &_b5]).unwrap();
//     // b3_b4_b5.reduce();
//     let mut b3_b4_b6 = max_phs(&vec![&_b3, &_b4, &_b6]).unwrap();
//     // b3_b4_b6.reduce();
//     // let mut b3_b5_b6 = max_phs(&vec![&_b3, &_b5, &_b6]).unwrap();
//     b3_b5_b6.reduce();
//     // let mut b4_b5_b6 = max_phs(&vec![&_b4, &_b5, &_b6]).unwrap();
//     b4_b5_b6.reduce();
//     let ands = vec![
//         // &b1_b2_b3, &b1_b2_b4, &b1_b2_b5, &b1_b2_b6, &b1_b3_b4, &b1_b3_b5, &b1_b3_b6, &b1_b4_b5,
//         // &b1_b4_b6, &b1_b5_b6, &b2_b3_b4, &b2_b3_b5, &b2_b3_b6, &b2_b4_b5, &b2_b4_b6, &b2_b5_b6,
//         &b3_b4_b5, &b3_b4_b6, //, &b3_b5_b6, &b4_b5_b6,
//     ]
//     let mut _top = min_phs(&ands).unwrap();
//     // _top.reduce();
//     _top.to_coxian().export("vot3of6")?;
//     Ok(())
// }
