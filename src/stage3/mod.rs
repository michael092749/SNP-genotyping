use crate::stage2::Translation;
use rayon::prelude::*;
use packed_simd::f64x2;

pub struct Rotation;

impl Rotation{

    pub fn rotate(data: &mut Vec<(f64, f64)>, offset_x: f64, offset_y: f64) -> f64 {
        // Correct for translation
        for point in data.iter_mut() {
            point.0 -= offset_x;
            point.1 -= offset_y;
        }
        
        // X-Sweep for Control Points
        let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
        let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);

        let control_points: Vec<(f64, f64)> = (0..400).map(|i| {
            let x = x_min + i as f64 * (x_max - x_min) / 399.0;
            data.iter().cloned().min_by_key(|&(x1, _)| (x1 - x).abs() as i64).unwrap()
        }).collect();
        
    
        // Fit a straight line to the control points
        let (m_control, _) = Translation::fit_line_p(&control_points);
    
        // Calculate the angle of rotation
        let theta = m_control.atan();
    
        theta
    }

pub fn rotate_p(data: &mut Vec<(f64, f64)>, offset_x: f64, offset_y: f64) -> f64 {
  // Prepare the SIMD offset vector - To utilize AVX2, you'll want to make use of the 256-bit wide SIMD registers. 
    let offset = f64x2::new(offset_x, offset_y);
    // Correct for translation using SIMD
    for point in data.iter_mut() {
        let original = f64x2::new(point.0, point.1);
        let corrected = original - offset;
        point.0 = corrected.extract(0);
        point.1 = corrected.extract(1);
    }

    // X-Sweep for Control Points
    let x_min = data.par_iter().map(|&(x, _)| x).reduce_with(f64::min).unwrap_or(f64::INFINITY);
    let x_max = data.par_iter().map(|&(x, _)| x).reduce_with(f64::max).unwrap_or(f64::NEG_INFINITY);

    let control_points: Vec<(f64, f64)> = (0..400).into_par_iter()
        .map(|i| {
            let x = x_min + i as f64 * (x_max - x_min) / 399.0;
            data.iter().cloned().min_by_key(|&(x1, _)| (x1 - x).abs() as i64).unwrap()
        })
        .collect();
    
    // Fit a straight line to the control points
    let (m_control, _) = Translation::fit_line_p(&control_points);
    
    // Calculate the angle of rotation
    let theta = m_control.atan();
    
    theta
}

    
}