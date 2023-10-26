use crate::stage2::Translation;
use rayon::prelude::*;
use packed_simd::f64x4;

pub struct Shear;

impl Shear {

    pub fn shear_unparallelised(data: &mut Vec<(f64, f64)>, theta: f64) -> f64 {
        // Correct for rotation
        for point in data.iter_mut() {
            let temp_x = point.0;
            let temp_y = point.1;

            point.0 = temp_x * theta.cos() + temp_y * theta.sin();
            point.1 = -temp_x * theta.sin() + temp_y * theta.cos();
        }

        // Y-Sweep for Control Points
        let y_min = data.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);
        let y_max = data.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, f64::max);
        let control_points: Vec<(f64, f64)> = (0..400).map(|i| {
            let y = y_min + i as f64 * (y_max - y_min) / 399.0;
            data.iter().cloned().min_by_key(|&(_, y1)| (y1 - y).abs() as i64).unwrap()
        }).collect();

        let (m_shear, _) = Translation::fit_line(&control_points);

        // The angle of this line identifies the shear parameter
        let shear_angle = m_shear.atan();

        shear_angle
    }

    pub fn shear_p(data: &mut Vec<(f64, f64)>, theta: f64) -> f64 {
        // Correct for rotation using AVX
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let cos_values = f64x4::new(cos_theta, cos_theta, cos_theta, cos_theta);
        let sin_values = f64x4::new(sin_theta, sin_theta, sin_theta, sin_theta);
    
        let chunks = data.chunks_mut(2);
        for chunk in chunks {
            if chunk.len() == 2 {
                let original = f64x4::new(chunk[0].0, chunk[0].1, chunk[1].0, chunk[1].1);
                let rotated = f64x4::new(
                    original.extract(0) * cos_values.extract(0) + original.extract(1) * sin_values.extract(0),
                    -original.extract(0) * sin_values.extract(0) + original.extract(1) * cos_values.extract(0),
                    original.extract(2) * cos_values.extract(2) + original.extract(3) * sin_values.extract(2),
                    -original.extract(2) * sin_values.extract(2) + original.extract(3) * cos_values.extract(2),
                );
                chunk[0].0 = rotated.extract(0);
                chunk[0].1 = rotated.extract(1);
                chunk[1].0 = rotated.extract(2);
                chunk[1].1 = rotated.extract(3);
            } else {
                // Handle the leftover point with scalar operations
                let temp_x = chunk[0].0;
                chunk[0].0 = temp_x * cos_theta + chunk[0].1 * sin_theta;
                chunk[0].1 = -temp_x * sin_theta + chunk[0].1 * cos_theta;
            }
        }
    
        // Y-Sweep for Control Points in parallel
        let y_min = data.par_iter().map(|&(_, y)| y).reduce_with(f64::min).unwrap_or(f64::INFINITY);
        let y_max = data.par_iter().map(|&(_, y)| y).reduce_with(f64::max).unwrap_or(f64::NEG_INFINITY);
        let control_points: Vec<(f64, f64)> = (0..400).into_par_iter()
        .map(|i| {
            let y = y_min + i as f64 * (y_max - y_min) / 399.0;
            data.par_iter().cloned().min_by_key(|&(_, y1)| (y1 - y).abs() as i64).unwrap()
        }).collect();
    
        let (m_shear, _) = Translation::fit_line_p(&control_points);
    
        let shear_angle = m_shear.atan();
    
        shear_angle
    }
}