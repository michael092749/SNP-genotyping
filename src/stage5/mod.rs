use rayon::prelude::*;
use packed_simd::f64x4;

pub struct Scale;

impl Scale {
    
    pub fn scale(data: &mut Vec<(f64, f64)>, shear_angle: f64)  {
    // Correct for shear
    for point in data.iter_mut() {
        let temp_x2 = point.0;
        let temp_y2 = point.1;
        point.0 = temp_x2 - shear_angle * temp_y2; // temp x3
        point.1 = temp_y2;                   // temp y3
    }

    // X-sweep for virtual points
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let x_virtual_points: Vec<f64> = (0..400).map(|i| {
        let x = x_min + i as f64 * (x_max - x_min) / 399.0;
        data.iter().cloned().min_by_key(|&(x1, _)| (x1 - x).abs() as i64).unwrap().0
    }).collect();

    // Robust mean for scale_x
    let scale_x = Self::robust_mean(&x_virtual_points);

    // Y-sweep for virtual points (triangulation)
    let y_min = data.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, f64::max);
    let y_virtual_points: Vec<f64> = (0..400).map(|i| {
        let y = y_min + i as f64 * (y_max - y_min) / 399.0;
        data.iter().cloned().min_by_key(|&(_, y1)| (y1 - y).abs() as i64).unwrap().1
    }).collect();

    // Robust mean for scale_y
    let scale_y = Self::parallel_mean(&y_virtual_points);
    
    data.par_iter_mut().for_each(|point| {
        point.0 /= scale_x; // x_n
        point.1 /= scale_y; // y_n
    });
}

pub fn robust_mean(values: &Vec<f64>) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn scale_p(data: &mut Vec<(f64, f64)>, shear_angle: f64) {
    // Correct for shear using AVX2
    let shear = shear_angle.tan();
    let shear_values = f64x4::splat(shear);

    data.par_chunks_mut(2).for_each(|chunk| {
        let original_x = f64x4::new(chunk[0].0, chunk[0].1, chunk.get(1).map_or(0.0, |p| p.0), chunk.get(1).map_or(0.0, |p| p.1));
        let corrected = original_x - shear_values * original_x;
        chunk[0].0 = corrected.extract(0);
        chunk[0].1 = corrected.extract(1);
        if let Some(point) = chunk.get_mut(1) {
            point.0 = corrected.extract(2);
            point.1 = corrected.extract(3);
        }
    });

    let x_min = data.par_iter().map(|&(x, _)| x).reduce_with(f64::min).unwrap_or(f64::INFINITY);
    let x_max = data.par_iter().map(|&(x, _)| x).reduce_with(f64::max).unwrap_or(f64::NEG_INFINITY);

    let x_virtual_points: Vec<f64> = (0..400).into_par_iter()
        .map(|i| {
            let x = x_min + i as f64 * (x_max - x_min) / 399.0;
            data.iter().cloned().min_by_key(|&(x1, _)| (x1 - x).abs() as i64).unwrap().1
        })
        .collect();

    let scale_x = Self::parallel_mean(&x_virtual_points);
            // Y-Sweep for Control Points in parallel
    let y_min = data.par_iter().map(|&(_, y)| y).reduce_with(f64::min).unwrap_or(f64::INFINITY);
    let y_max = data.par_iter().map(|&(_, y)| y).reduce_with(f64::max).unwrap_or(f64::NEG_INFINITY);
    let y_virtual_points: Vec<f64> = (0..400).into_par_iter()
    .map(|i| {
        let y = y_min + i as f64 * (y_max - y_min) / 399.0;
        data.par_iter().cloned().min_by_key(|&(_, y1)| (y1 - y).abs() as i64).unwrap().1
    }).collect();

    let scale_y = Self::parallel_mean(&y_virtual_points);

    let scale_values = f64x4::new(scale_x, scale_y, scale_x, scale_y);
    data.par_chunks_mut(2).for_each(|chunk| {
        let original = f64x4::new(chunk[0].0, chunk[0].1, chunk.get(1).map_or(0.0, |p| p.0), chunk.get(1).map_or(0.0, |p| p.1));
        let scaled = original / scale_values;
        chunk[0].0 = scaled.extract(0);
        chunk[0].1 = scaled.extract(1);
        if let Some(point) = chunk.get_mut(1) {
            point.0 = scaled.extract(2);
            point.1 = scaled.extract(3);
        }
    });

}

pub fn parallel_mean(values: &Vec<f64>) -> f64 {
    let sum: f64 = values.par_iter().sum();
    sum / values.len() as f64
}

}


