use rayon::prelude::*;


pub struct Outliers;

impl Outliers{

    pub fn remove_outliers_parallelised(data: &mut Vec<(f64, f64)>) -> Result<Vec<(usize, (f64, f64))>, i32> {

        let mut x_values: Vec<f64> = Vec::new();
        let mut y_values: Vec<f64> = Vec::new();
        let mut ratios: Vec<f64> = Vec::new();
    
        for &(x, y) in data.iter() {
            x_values.push(x);
            y_values.push(y);
            ratios.push(x / (x + y));
        }
    
        x_values.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        y_values.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));    
        ratios.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let get_thresholds = |values: &Vec<f64>| {
            let len = values.len();
            let fifth_smallest = values[4];
            let fifth_largest    = values[len - 5];
            let first_percentile = values[(0.01 * len as f64) as usize];
            let ninety_ninth_percentile = values[(0.99 * len as f64) as usize];
            (
                f64::min(fifth_smallest, first_percentile),
                f64::max(fifth_largest, ninety_ninth_percentile),
            )
        };
        
        let (x_min, x_max) = get_thresholds(&x_values);
        let (y_min, y_max) = get_thresholds(&y_values);
        let (ratio_min, ratio_max) = get_thresholds(&ratios);

        let mut outliers = Vec::new();
        let mut i = 0;
        while i < data.len() {
            let (x, y) = data[i];
            let ratio = x / (x + y);
            if x <= x_min || x >= x_max || y <= y_min || y >= y_max || ratio <= ratio_min || ratio >= ratio_max {
                outliers.push((i, (x, y)));
                data.remove(i);
            } else {
                i += 1;
            }
        }

        if data.len() == 0 {
            return Err(-1);
        }

        Ok(outliers)
    }   



    pub fn remove_outliers(data: &mut Vec<(f64,f64)>) {
        let mut x_values: Vec<f64> = Vec::with_capacity(data.len());
        let mut y_values: Vec<f64> = Vec::with_capacity(data.len());
        let mut ratios: Vec<f64> = Vec::with_capacity(data.len());

        for &(x, y) in data.iter() {
            x_values.push(x);
            y_values.push(y);
            ratios.push(x / (x + y));
        }

        x_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let get_thresholds = |values: &Vec<f64>| {
            let len = values.len();
            let fifth_smallest = values[4];
            let fifth_largest = values[len - 5];
            let first_percentile = values[(0.01 * len as f64) as usize];
            let ninety_ninth_percentile = values[(0.99 * len as f64) as usize];
            (
                f64::min(fifth_smallest, first_percentile),
                f64::max(fifth_largest, ninety_ninth_percentile),
            )
        };

        let (x_min, x_max) = get_thresholds(&x_values);
        let (y_min, y_max) = get_thresholds(&y_values);
        let (ratio_min, ratio_max) = get_thresholds(&ratios);

        data.retain(|&(x, y)| {
            let ratio = x / (x + y);
            x > x_min && x < x_max && y > y_min && y < y_max && ratio > ratio_min && ratio < ratio_max
        });

    }

}