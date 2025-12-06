#[cfg(test)]
/// True values taken from Storm-dft. Using Rationals is to precise for the floating point used by storm, thus, we have to have a large threshold instead of using 1 ULP.
mod tests {

    mod basic {
        use aph::{
            formats::ft::{aph_from_ft, utils::ConstructionMethod},
            linalg::fields::rational::Rational,
        };
        #[test]
        fn and() {
            let source = String::from(
                "toplevel \"top\";\n\"top\" and \"a1\" \"a2\";\n\"a1\" lambda=1;\n\"a2\" lambda=2 phases=2;\n\"a3\" lambda=3 phases=3;\n",
            );

            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            assert!(f64::abs(1.444444444 - mttf) < 1e-6);
        }

        #[test]
        fn vot2oof4() {
            let source = String::from(
                "toplevel \"System\";\n\"System\" 2of4 \"g1\" \"g2\" \"a5\" \"a6\";\n\"g1\" and \"a1\" \"a2\";\n\"g2\" and \"a3\" \"a4\";\n\"a1\" lambda=1 phases=2 dorm=0;\n\"a2\" lambda=2 dorm=0;\n\"a3\" lambda=3 phases=2 dorm=0;\n\"a4\" lambda=4 phases=2 dorm=0;\n\"a5\" lambda=5 dorm=0;\n\"a6\" lambda=6 dorm=0;",
            );

            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            assert!(f64::abs(0.2535546458 - mttf) < 1e-6);
        }

        #[test]
        fn or() {
            let source = String::from(
                "toplevel \"top\";\n\"top\" or \"a1\" \"a2\" \"a3\";\n\"a1\" lambda=1 phases=3 dorm=0;\n\"a2\" lambda=2 phases=2 dorm=0;\n\"a3\" lambda=3 phases=3 dorm=0;\n",
            );
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            assert!(f64::abs(0.6388888889 - mttf) < 1e-6);
        }
    }

    mod spares {
        use aph::{
            formats::ft::{aph_from_ft, utils::ConstructionMethod},
            linalg::fields::rational::Rational,
        };
        #[test]
        fn csp_and_2sh() {
            let source = std::fs::read_to_string("src/tests/csp_and_2sh.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 5.067566491;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
        #[test]
        fn csp_gates_and() {
            let source = std::fs::read_to_string("src/tests/csp_gates_and.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 6.147435897;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
        #[test]
        fn csp_gates_or() {
            let source = std::fs::read_to_string("src/tests/csp_gates_or.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 1.185897436;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
        #[test]
        fn csp_mix_shared() {
            let source = std::fs::read_to_string("src/tests/csp_mix_shared.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 15.57519292;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
        #[test]
        fn csp_or_2sh() {
            let source = std::fs::read_to_string("src/tests/csp_or_2sh.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 2.265766842;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
        #[test]
        fn hsp() {
            let source = std::fs::read_to_string("src/tests/hsp.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 11.69623688;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }

        #[test]
        fn wsp() {
            let source = std::fs::read_to_string("src/tests/wsp.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 1.053282828;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
    }

    mod fdep {
        use aph::{
            formats::ft::{aph_from_ft, utils::ConstructionMethod},
            linalg::fields::rational::Rational,
        };
        #[test]
        fn fdep_gate() {
            let source = std::fs::read_to_string("src/tests/fdep_gate.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 0.4876800946;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }

        #[test]
        fn fdep_csp_cuc() {
            let source = std::fs::read_to_string("src/tests/fdep_csp_cuc.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 0.5909090909;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }

        #[test]
        fn fdep_csp_sp() {
            let source = std::fs::read_to_string("src/tests/fdep_csp_sp.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 1.007575758;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }

        #[test]
        fn fdep_circ() {
            let source = std::fs::read_to_string("src/tests/fdep_circ.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                true, // Try to modularise, analyse the behaviour of a circular fdep.
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 1.170201713;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
    }

    mod seq {
        use aph::{
            formats::ft::{aph_from_ft, utils::ConstructionMethod},
            linalg::fields::rational::Rational,
        };
        #[test]
        fn seq() {
            let source = std::fs::read_to_string("src/tests/seq.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 1.84047619;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
        #[test]
        fn seq_fdep() {
            let source = std::fs::read_to_string("src/tests/seq_fdep.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());
            let true_val = 1.839482903;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
        #[test]
        fn seq_gate() {
            let source = std::fs::read_to_string("src/tests/seq_gate.dft").unwrap();
            let aph = aph_from_ft::<Rational>(
                &source,
                true,
                false,
                ConstructionMethod::Sparse,
                aph::linalg::fields::Round::Nearest,
                aph::formats::ft::utils::RoundMode::Mix,
            );
            let mttf = f64::from(aph.expected_value().unwrap());

            let true_val = 1.227777778;
            assert!(
                f64::abs(true_val - mttf) < 1e-6,
                "(True) {true_val} vs {mttf} (APH)"
            );
        }
    }
}
