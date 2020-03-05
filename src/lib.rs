extern crate rand;

// for logging (debug mostly, switched at compile time in cargo.toml)
extern crate log;
extern crate simple_logger;

#[macro_use]
extern crate lazy_static;



lazy_static! {
    #[allow(dead_code)]
    static ref LOG: u64 = {
        let res = init_log();
        res
    };
}

// install a logger facility
fn init_log() -> u64 {
    simple_logger::init().unwrap();
    println!("\n ************** initializing logger *****************\n");    
    return 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    // initialize once log system for tests.
    fn init_log() {
        let _res = simple_logger::init();
    }
}  // end of tests
