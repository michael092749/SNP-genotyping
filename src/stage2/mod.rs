use rayon::prelude::*;
// use faster::prelude::*;
// use std::thread;
// use crossbeam;
pub struct Translation;
use nalgebra::base::DMatrix;
// use nalgebra::linalg::SVD;
use std::cmp::Ordering;

impl Translation{

pub fn transform(data: &mut Vec<(f64, f64)>) -> (f64, f64) {

    // Sample 400 points along the x-axis and y-axis
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let y_min = data.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, f64::max);

    let x_step = (x_max - x_min) / 399.0;
    let y_step = (y_max - y_min) / 399.0;

    let x_samples: Vec<f64> = (0..400).map(|i| x_min + i as f64 * x_step).collect();
    let y_samples: Vec<f64> = (0..400).map(|i| y_min + i as f64 * y_step).collect();
    
    // Find the closest SNP to each sampled point using the external function
    let homozygote_a: Vec<(f64, f64)> = x_samples.iter().map(|&x| Self::find_closest(x, 'x', &data)).collect();
    let homozygote_b: Vec<(f64, f64)> = y_samples.iter().map(|&y| Self::find_closest(y, 'y', &data)).collect();

    // Fit a straight line to the candidate homozygote A alleles and homozygote B alleles using the external function
    let (m_a, c_a) = Self::fit_line(&homozygote_a);
    let (m_b, c_b) = Self::fit_line(&homozygote_b);

    // Compute the intercept of the two lines
    let offset_x = (c_b - c_a) / (m_a - m_b);
    let offset_y = m_a * offset_x + c_a;

    (offset_x, offset_y)
}
        
pub fn transform_p(data: &mut Vec<(f64, f64)>) -> (f64, f64) {

    let (x_min, x_max, y_min, y_max) = data.par_iter().fold(
        || (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        |(xmin, xmax, ymin, ymax), &(x, y)| {
            (xmin.min(x), xmax.max(x), ymin.min(y), ymax.max(y))
        },
    ).reduce(
        || (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        |(xmin_a, xmax_a, ymin_a, ymax_a), (xmin_b, xmax_b, ymin_b, ymax_b)| {
            (xmin_a.min(xmin_b), xmax_a.max(xmax_b), ymin_a.min(ymin_b), ymax_a.max(ymax_b))
        },
    );

    let x_step = (x_max - x_min) / 399.0;
    let y_step = (y_max - y_min) / 399.0;

    let x_samples: Vec<f64> = (0..400).map(|i| x_min + i as f64 * x_step).collect();
    let y_samples: Vec<f64> = (0..400).map(|i| y_min + i as f64 * y_step).collect();

    let homozygote_a: Vec<(f64, f64)> = x_samples.par_iter().map(|&x| Self::find_closest(x, 'x', &data)).collect();
    let homozygote_b: Vec<(f64, f64)> = y_samples.par_iter().map(|&y| Self::find_closest(y, 'y', &data)).collect();

    let (m_a, c_a) = Self::fit_line_p(&homozygote_a);
    let (m_b, c_b) = Self::fit_line_p(&homozygote_b);

    let offset_x = (c_b - c_a) / (m_a - m_b);
    let offset_y = m_a * offset_x + c_a;

    (offset_x, offset_y)
}



pub fn find_closest(point: f64, axis: char, data: &Vec<(f64, f64)>) -> (f64, f64) {
    data.iter().cloned().min_by(|&(x1, y1), &(x2, y2)| {
        let dist1 = if axis == 'x' { (x1 - point).abs() } else { (y1 - point).abs() };
        let dist2 = if axis == 'x' { (x2 - point).abs() } else { (y2 - point).abs() };
        dist1.partial_cmp(&dist2).unwrap_or(Ordering::Equal)
    }).unwrap_or_else(|| {
        println!("No points found to compare, returning (0.0, 0.0) as a fallback.");
        (0.0, 0.0)
    })
}

pub fn fit_line(points: &Vec<(f64, f64)>) -> (f64, f64) {
    let matrix = DMatrix::from_iterator(points.len(), 2, points.iter().map(|&(x, _)| vec![x, 1.0].into_iter()).flatten());
    let b = DMatrix::from_column_slice(points.len(), 1, &points.iter().map(|&(_, y)| y).collect::<Vec<_>>());
    let svd = matrix.svd(true, true);
    let solution = svd.solve(&b, 1.0e-10).unwrap();
    (solution[(0, 0)], solution[(1, 0)])
}

pub fn fit_line_p(points: &Vec<(f64, f64)>) -> (f64, f64) {
    let matrix_data: Vec<f64> = points.par_iter()
        .flat_map(|&(x, _)| vec![x, 1.0])
        .collect();

    let matrix = DMatrix::from_vec(points.len(), 2, matrix_data);

    let b_data: Vec<f64> = points.par_iter()
        .map(|&(_, y)| y)
        .collect();

    let b = DMatrix::from_column_slice(points.len(), 1, &b_data);

    let svd = matrix.svd(true, true);
    let solution = svd.solve(&b, 1.0e-10).unwrap();
    (solution[(0, 0)], solution[(1, 0)])
}
}
