//! To read MNIST database

use std::io::prelude::*;

fn readImageFile(io_in: &mut dyn Read) {
    // read 4 bytes magic
    let magic : u32;
    // to read 32 bits in network order!
    let mut toread : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    magic = u32::from_be(toread);
    // read nbitems
    let nbitem : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbitem = u32::from_be(toread);
    //  read nbrow
    let nbrow : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbrow = u32::from_be(toread);    
    // read nbcolumns
    let nbcolum : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbcolum = u32::from_be(toread);     
    // for each item, read a row of nbcolumns u8
} // end of readImageFile