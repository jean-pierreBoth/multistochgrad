//! To read MNIST database

use std::io::prelude::*;
use std::io::{BufReader};
use ndarray::{Array3, Array1, s};
use std::fs::{OpenOptions};

/// A struct to load/store http://yann.lecun.com/exdb/mnist/
/// train-labels-idx1-ubyte  digits between 0 and 9
/// ans hand written characters as 28*28 images with values between 0 and 255 train-images-idx3-ubyte
pub struct MnistData {
    _image_filename : String,
    _label_filename : String,
    images : Array3::<u8>,
    labels : Array1::<u8>,
}


impl MnistData {
    pub fn new(image_filename : String, label_filename : String) -> std::io::Result<MnistData> {
        let image_file = OpenOptions::new().read(true).open(&image_filename)?;
        let mut image_io = BufReader::new(image_file);
        let images = read_image_file(&mut image_io);
        // labels
        let labels_file = OpenOptions::new().read(true).open(&label_filename)?;
        let mut labels_io = BufReader::new(labels_file);
        let labels = read_label_file(&mut labels_io);
        Ok(MnistData{
            _image_filename : image_filename,
            _label_filename : label_filename,
            images,
            labels
        } )
    } // end of new for MnistData


} // end of impl MnistData


pub fn read_image_file(io_in: &mut dyn Read) -> Array3::<u8> {
    // read 4 bytes magic
    let magic : u32;
    // to read 32 bits in network order!
    let toread : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    magic = u32::from_be(toread);
    assert_eq!(magic, 2051);
    // read nbitems
    let nbitem : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbitem = u32::from_be(toread);
    assert_eq!(nbitem, 60000);
    //  read nbrow
    let nbrow : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbrow = u32::from_be(toread); 
    assert_eq!(nbrow, 255);   
    // read nbcolumns
    let nbcolumn : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbcolumn = u32::from_be(toread);     
    assert_eq!(nbcolumn,255);   
    // for each item, read a row of nbcolumns u8
    let mut images = Array3::<u8>::zeros((3, 4, nbitem as usize));
    let mut datarow = Vec::<u8>::with_capacity(nbcolumn as usize);
    for k in 0..nbitem as usize {
        for i in 0..nbrow as usize {
            let it_slice ;
            it_slice = datarow.as_mut_slice();
            io_in.read_exact(it_slice).unwrap();
            let mut smut_ik = images.slice_mut(s![i, .., k]);
            for j in 0..smut_ik.len() {
                smut_ik[j] = it_slice[j];
            }
        //    for j in 0..nbcolumn as usize {
        //        *(images.get_mut([i,j,k]).unwrap()) = it_slice[j];
        //   }            
            // how do a block copy from read slice to view of images.
           // images.slice_mut(s![i as usize, .. , k as usize]).assign(&Array::from(it_slice)) ;  

        }
    }
    images
} // end of readImageFile



pub fn read_label_file(io_in: &mut dyn Read) -> Array1<u8>{
    let magic : u32;
    // to read 32 bits in network order!
    let toread : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    magic = u32::from_be(toread);
    assert_eq!(magic, 2049);
     // read nbitems
     let nbitem : u32;
     let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
     io_in.read_exact(it_slice).unwrap();
     nbitem = u32::from_be(toread);   
     assert_eq!(nbitem, 60000);
     let mut labels_vec = Vec::<u8>::with_capacity(nbitem as usize);
     io_in.read_exact(&mut labels_vec).unwrap();
     let labels = Array1::from(labels_vec);
     labels
    }  // end of fn read_label