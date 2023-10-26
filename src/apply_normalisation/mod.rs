use crate::stage1::Outliers;
use crate::stage2::Translation;
use crate::stage3::Rotation;
use crate::stage4::Shear;
use crate::stage5::Scale;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

pub struct Normalise;

impl Normalise {

    pub fn within_beadset_normalisation(data: &mut HashMap<i32, Vec<(f64,f64)>>, vector_names: &Arc<Mutex<Vec<i32>>>) -> Result<(), Box<dyn Error>> {

        //let mut data = beadset_id_vector.lock().unwrap();
        let vector_names = vector_names.lock().unwrap();

        for &name in &*vector_names {
            if let Some(data_vector) = data.get_mut(&name) {
                // Stage 1 - remove outliers
                if let Ok(removed_outliers) = Outliers::remove_outliers_parallelised(data_vector) {
                        // // Stage 2 - Translation
                        let (offset_x, offset_y) = Translation::transform_p(data_vector);

                        // // Stage 3 - Rotation
                        let theta:f64 = Rotation::rotate_p(data_vector,offset_x,offset_y);

                        // // Stage 4
                        let shear = Shear::shear_p(data_vector,theta);

                        // // Stage 5
                        Scale::scale_p(data_vector,shear);

                        for (index, outlier) in removed_outliers {
                            data_vector.insert(index, outlier);
                        }
                } else {
                    println!("Outliers function returned an error, skipping transform_p and other functions.");
                }
            }
        }

        Ok(())
    }


    pub fn within_snp_normalisation(beadset_id_vector: &Arc<Mutex<Vec<(f64,f64)>>>) -> Result<(), Box<dyn Error>> {
        let mut data = beadset_id_vector.lock().unwrap();

        // Stage 1 - remove outliers
        if let Ok(removed_outliers) = Outliers::remove_outliers_parallelised(&mut data) {

            // Stage 2 - Translation
            let (offset_x, offset_y) = Translation::transform_p(&mut data);

            // Stage 3 - Rotation
            let theta:f64 = Rotation::rotate_p(&mut data,offset_x,offset_y);

            // Stage 4
            let shear = Shear::shear_p(&mut data,theta);

            // Stage 5
            Scale::scale_p(&mut data,shear);

            for (index, outlier) in removed_outliers {
                data.insert(index, outlier);
            }

        } else {
            println!("Outliers function returned an error, skipping transform_p and other_function.");
        }

        Ok(())
    }
}
