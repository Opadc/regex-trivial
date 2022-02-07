#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // fuzzed code goes here
    if let Ok(data) = std::str::from_utf8(data) {
       let _ =  regex::Regex::new(data);
    }
});
