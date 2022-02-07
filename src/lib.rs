//! regex - Regular expression pattern matching and replacement
//!
//! ## Usage
//! see regex.rs test mod
//!
//!
//! ## Regular Expressions:
//!
//!     char        match itself, emit code 'EXACTLY char'
//!         
//!      .          match any character. emit code 'ANY'
//!
//!      \          matches the character following it. emit code 'EXACTLY char'
//!
//!     [set]       matches one of the characters in the set.
//!                 If the first character in the set is "^",
//!                 it matches a character NOT in the set,
//!                 emit code 'ANYOF set' or "ANYBUT set"
//!
//!       *         any regular expression followed by closure char (*) matches
//!                 zero or more matches of that form.
//!                 emit code "STAR char" for simple case or 'TODO:'
//!
//!       +         same as '*', except it matches one or more.
//!                 emit code "PLUS char" for simple case or 'TODO:'
//!     
//!       ?         zero or one. emit code 'TODO:  '
//!
//!       ^         match the beginning of the line. emit code "BOL"
//!
//!       $         match the end of line. emit code "EOL"
//!                   
//! ## Details
//!
//! ## Code emit example
//!     '()' contain the position of next node
//! ```plain
//!     regex_compile("a")
//!     result :
//!         BEGIN
//!             1:BRANCH(15)
//!             4:EXACTLY(15)a
//!             15:END(0)
//!                            
//!     regex_compile("a*")  
//!     result :
//!         BEGIN
//!             1:BRANCH(18)
//!             4:STAR(18)
//!             7:EXACTLY(0)a   
//!             18:END(0)
//!
//!
//!     regex_compile("a?")  
//!     result :
//!         BEGIN
//!             1:BRANCH(24)
//!             4:BRANCH(18)    
//!             7:EXACTLY(21)a
//!             18:BRANCH(21)
//!             21:NOTHING(24)  
//!             24:END(0)
//!
//! ```
//!
//!
#![feature(iter_advance_by)]
#[macro_use]
extern crate enum_display_derive;

mod regex;

pub use regex::Regex;