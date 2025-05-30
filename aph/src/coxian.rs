use std::{
    fs::File,
    io::{self, Write},
};

use log::info;

use crate::linalg::{Vector, fields::PseudoField};

/// A Coxian distribution, is a special type of PH where the generator is denored by $\mathbf{Cx}(\[\lambda_1, p_2\],\[\lambda_1, p_2\],\dots,\lambda_n)$, and
/// has the form:
/// /// $$
/// \begin{bmatrix}
/// -\lambda_1  &  p1\lambda_1  & 0            & \dots  & 0       \\
/// 0           &  -\lambda_2   & p2\lambda_2  & \dots  & 0       \\
/// \vdots      &  \vdots       & \vdots       & \ddots & \vdots  \\
/// 0           &  0            & 0            & \dots  & -\lambda_n
/// \end{bmatrix}
/// $$
#[derive(Debug)]
pub struct Coxian<F: PseudoField> {
    pub lambdas: Vector<F>,
    pub factors: Vector<F>,
}

impl<F: PseudoField> Coxian<F> {
    pub fn tra_lab_export(&self, filename: &str) -> io::Result<()> {
        let mut tra_file = File::create(format!("{:?}.tra", filename).replace("\"", ""))?;
        let mut lab_file = File::create(format!("{:?}.lab", filename).replace("\"", ""))?;

        let size = self.lambdas.len();

        writeln!(tra_file, "ctmc")?;

        // There is no need of writing the initial distribution.

        // Write transitions, i.e. each state to the next and to the last one.
        for i in 0..size {
            let mut l = self.lambdas[i].clone();
            l.neg_assign();
            let to_next = l.clone() * self.factors[i].clone(); // to the next state
            l.sub_assign(&to_next); // to the final state

            if !to_next.is_zero() {
                writeln!(tra_file, "{} {} {}", i, i + 1, to_next.to_string())?;
            }
            if i + 2 != size + 1 {
                writeln!(tra_file, "{} {} {}", i, size, l.to_string())?;
            }
        }

        // Make last state absorbing
        writeln!(tra_file, "{} {} 1", size, size)?;

        // Write label file for storm.
        writeln!(lab_file, "#DECLARATION")?;
        writeln!(lab_file, "init done")?;
        writeln!(lab_file, "#END")?;
        writeln!(lab_file, "0 init")?;
        writeln!(lab_file, "{} done", size)?;

        info!(
            "Coxian Model writed to file: '{}', last state with label: 'done'.",
            filename
        );

        Ok(())
    }

    /// Exports to prism compatible CTMCs
    pub fn ctmc_export(&self, filename: &str) -> io::Result<()> {
        let mut ctmc_file = File::create(format!("{:?}.ctmc", filename).replace("\"", ""))?;

        let size = self.lambdas.len();

        writeln!(ctmc_file, "ctmc")?;
        writeln!(ctmc_file, "module main")?;
        writeln!(ctmc_file, "s: [0..{}] init 0;", size)?;

        // There is no need of writing the initial distribution.

        // Write transitions, i.e. each state to the next and to the last one.
        for i in 0..size {
            let mut to_abs = self.lambdas[i].clone();
            to_abs.neg_assign();
            let to_next = to_abs.clone() * self.factors[i].clone(); // to the next state
            to_abs.sub_assign(&to_next); // to the final state

            if i + 1 != size {
                writeln!(
                    ctmc_file,
                    "<> s={} -> {} : (s'={}) + {} : (s'={});",
                    i,
                    to_next.to_string(),
                    i + 1,
                    to_abs.to_string(),
                    size,
                )?;
            } else {
                writeln!(
                    ctmc_file,
                    "<> s={} -> {} : (s'={});",
                    i,
                    to_next.to_string(),
                    i + 1,
                )?;
            }
        }

        // writeln!(ctmc_file, "<> s={} -> {} : (s'={});", size, 0.0, size)?;

        writeln!(ctmc_file, "endmodule\n")?;
        writeln!(ctmc_file, "label \"done\"= (s={});", size)?;

        info!(
            "Coxian Model writed to file: '{}', last state with label: 'done'.",
            filename
        );

        Ok(())
    }
}
