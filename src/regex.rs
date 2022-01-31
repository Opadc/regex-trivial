#![allow(clippy::upper_case_acronyms)]

use bitflags::bitflags;
use itertools::{peek_nth, PeekNth};
use num_enum::{IntoPrimitive, TryFromPrimitive};

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Sub},
    str::Chars,
};

const MAX_SUB_EXP: usize = 9;
const META_CHAR: & str = "^$.[()|?+*\\";
#[derive(Debug, Clone, PartialEq, Eq)]
enum Error {
    OutOfBound,
    InvalidOpcode,
    InvaildOperand,
    InvalidRegex(usize, &'static str),
    ExecuteFailed(PC, &'static str),
}

// directly copy from c.(
#[derive(Debug, IntoPrimitive, TryFromPrimitive, PartialEq, Eq, Display)]
#[repr(u8)]
enum Opcode {
    /* definition	number	opnd?	meaning */
    END = 255,   /* no	End of program. */
    BOL = 1,     /* no	Match beginning of line. */
    EOL = 2,     /* no	Match end of line. */
    ANY = 3,     /* no	Match any character. */
    ANYOF = 4,   /* str	Match any of these. */
    ANYBUT = 5,  /* str	Match any but one of these. */
    BRANCH = 6,  /* node	Match this, or the next..\&. */
    BACK = 7,    /* no	"next" ptr points backward. */
    EXACTLY = 8, /* str	Match this string. */
    NOTHING = 9, /* no	Match empty string. */
    STAR = 10,   /* node	Match this 0 or more times. */
    PLUS = 11,   /* node	Match this 1 or more times. */
    OPEN = 20,   /* no	Sub-RE starts here. */

    /* Strong Type ! */
    OPEN1 = 21,
    OPEN2 = 22,
    OPEN3 = 23,
    OPEN4 = 24,
    OPEN5 = 25,
    OPEN6 = 26,
    OPEN7 = 27,
    OPEN8 = 28,
    OPEN9 = 29,
    CLOSE = 30, /* no	Analogous to OPEN. */
    CLOSE1 = 31,
    CLOSE2 = 32,
    CLOSE3 = 33,
    CLOSE4 = 34,
    CLOSE5 = 35,
    CLOSE6 = 36,
    CLOSE7 = 37,
    CLOSE8 = 38,
    CLOSE9 = 39,
    BEGIN = 100, /*no   Promgram begin*/
}
impl Opcode {
    pub fn open_n(n: usize) -> Opcode {
        (Opcode::OPEN as u8 + n as u8).try_into().unwrap()
    }
    pub fn close_n(n: usize) -> Opcode {
        (Opcode::CLOSE as u8 + n as u8).try_into().unwrap()
    }
}

// use pc(index in to bincode vec) to simluate poiner
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PragramCounter(pub usize);
impl Display for PragramCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl PragramCounter {
    // PC(0) always point to Opcode::BEGIN
    const NULL: PragramCounter = PragramCounter(0);
}
impl Add<usize> for PragramCounter {
    type Output = PragramCounter;

    fn add(self, rhs: usize) -> Self::Output {
        PragramCounter(self.0 + rhs)
    }
}
impl AddAssign<usize> for PragramCounter {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}
impl Sub<usize> for PragramCounter {
    type Output = PragramCounter;

    fn sub(self, rhs: usize) -> Self::Output {
        PragramCounter(self.0 - rhs)
    }
}
type PC = PragramCounter;

// Varible contain compiled regex.
#[derive(Debug, Clone)]
struct Program {
    pub pc: PC,
    bincode: Vec<u8>,
}
impl Program {
    pub fn new() -> Program {
        Program {
            pc: PragramCounter(0),
            bincode: vec![],
        }
    }
    pub fn reserve(&mut self, additional: usize) {
        self.bincode.reserve(additional);
    }
    fn is_node_out_of_bound(&self, pc: PC) -> Result<(), Error> {
        if pc + 2 >= self.pc {
            Err(Error::OutOfBound)
        } else {
            Ok(())
        }
    }
    // return the opcode at the position pc in bincode.
    pub fn opcode_at(&self, pc: PC) -> Result<Opcode, Error> {
        self.is_node_out_of_bound(pc)?;
        Opcode::try_from(self.bincode[pc.0]).map_err(|_| Error::InvalidOpcode)
    }
    // return the next pc at the position pc in bincode.
    pub fn next_at(&self, pc: PC) -> Option<PC> {
        self.is_node_out_of_bound(pc).ok()?;
        let high_byte = self.bincode[(pc + 1).0] as usize;
        let low_byte = self.bincode[(pc + 2).0] as usize;
        let offset = ((high_byte & 0o177) << 8) + low_byte;

        if offset == 0 {
            return None;
        }
        if self.opcode_at(pc).ok()? == Opcode::BACK {
            Some(pc - offset)
        } else {
            Some(pc + offset)
        }
    }

    pub fn operand_at(&self, pc: PC) -> Option<PC> {
        self.is_node_out_of_bound(pc).ok()?;
        self.is_node_out_of_bound(pc + 3).ok()?;
        Some(pc + 3)
    }

    // push a new node in bincode, return origial pc before emit.
    pub fn emit_node(&mut self, opcode: Opcode) -> PC {
        let old_pc = self.pc;
        self.bincode.push(opcode.into());
        self.bincode.push(0);
        self.bincode.push(0);
        self.pc += 3;
        old_pc
    }
    pub fn emit_byte(&mut self, byte: u8) {
        self.bincode.push(byte);
        self.pc += 1;
    }

    // insert an operator in front of already-emitted operand
    // Means relocating the operand.
    pub fn insert_node(&mut self, opcode: Opcode, opnd: PC) -> Result<(), Error> {
        self.is_node_out_of_bound(opnd)?;
        self.bincode.push(0);
        self.bincode.push(0);
        self.bincode.push(0);
        let begin = opnd.0;
        for i in (begin + 3..self.bincode.len()).rev() {
            self.bincode[i] = self.bincode[i - 3];
        }

        self.bincode[begin] = opcode.into();
        self.bincode[begin + 1] = 0;
        self.bincode[begin + 2] = 0;
        self.pc += 3;
        Ok(())
    }

    // set the next-pointer at the end of a node chain
    pub fn update_tail(&mut self, mut chain: PC, new_tail: PC) -> Result<(), Error> {
        self.is_node_out_of_bound(chain)?;
        while let Some(next) = self.next_at(chain) {
            chain = next;
        }
        let offset = if self.opcode_at(chain)? == Opcode::BACK {
            chain.0 - new_tail.0
        } else {
            new_tail.0 - chain.0
        };
        self.bincode[chain.0 + 1] = ((offset >> 8) & 0o177) as u8;
        self.bincode[chain.0 + 2] = (offset & 0o377) as u8;
        Ok(())
    }

    // regtail on operand of first argument; nop if operandless
    pub fn update_oprand_tail(&mut self, branch: PC, new_tail: PC) -> Result<(), Error> {
        if self.opcode_at(branch)? != Opcode::BRANCH {
            Ok(())
        } else {
            self.update_tail(
                self.operand_at(branch).ok_or(Error::InvaildOperand)?,
                new_tail,
            )
        }
    }
}
impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", Opcode::BEGIN)?;
        let mut pc = PragramCounter(1);
        while let Ok(opcode) = self.opcode_at(pc) {
            write!(f, "{:>3}:{}", pc.0, opcode)?;
            if let Some(next) = self.next_at(pc) {
                if opcode == Opcode::BACK{
                    write!(f, "({})",pc.0 - (pc.0 - next.0))?;
                }else{
                    write!(f, "({})", pc.0 + (next.0 - pc.0))?;
                }
                
            } else {
                write!(f, "(0)")?;
            }
            pc += 3;
            if opcode == Opcode::ANY || opcode == Opcode::ANYBUT || opcode == Opcode::EXACTLY {
                let mut base = pc.0;
                while base + 4 < self.pc.0 {
                    let mut digit = [0u8; 4];
                    digit.copy_from_slice(&self.bincode[base..base + 4]);
                    pc += 4;
                    base = pc.0;
                    let ch = u32::from_be_bytes(digit);
                    if ch == 0 {
                        break;
                    }
                    let ch = char::try_from(ch).unwrap();
                    write!(f, "{}", ch)?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

fn is_repn(ch: char) -> bool {
    ch == '?' || ch == '*' || ch == '+'
}

fn is_mata(ch: char) -> bool {
    META_CHAR.find(ch).is_some()
}

pub struct Regexp{
    startp: [PC; MAX_SUB_EXP],
    endp:   [PC; MAX_SUB_EXP],
    // below is used to optimize
    regstart: char,
    reganch: char,
    regmust: String,
    program: Program,
}
impl Regexp{
    fn new(comp: Comp)->Regexp{
        Regexp { startp: [PC::NULL; MAX_SUB_EXP], endp: [PC::NULL; MAX_SUB_EXP], regstart: '\0', reganch: '\0', regmust: String::new(), program: comp.program() }
    }
}
//
bitflags! {
    pub struct CompStatus: u8{
        const WORST        =0;
        const HASWIDTH     =1;          // does sub match have length
        const SIMPLE       =2;          // does sub match enough simple to use STAR/PLUS operand, eg. 'a', 'b'.
        const SPSTART      =4;          // does sub match starts with * or +
    }
}
// Work-variable struct for regex compile
struct Comp<'a> {
    regparse: PeekNth<Chars<'a>>, // regex that need compile
    regposi: usize,               // char have looked in regparse
    paren_count: usize,           // () counts
    program: Program,             // compile result. byte code represent of nfa.
    emit_code: bool,              // if false, don't emit code.
    code_size: usize,
}

impl<'a> Comp<'a> {
    pub fn program(self)->Program{
        self.program
    }

    /*
    -   regcomp  compile a regular expression into internal code
    *
    *   compile regex will be two phase process. first for verify the validity of the regex, don't emit code.
    *
    */
    pub fn regcomp(exp: &str) -> Result<Regexp, Error> {
        let mut comp = Comp {
            regparse: peek_nth(exp.chars()),
            regposi: 0,
            paren_count: 1,
            program: Program::new(),
            emit_code: false,
            code_size: 0,
        };
        comp.regc(Opcode::BEGIN.into());
        comp.reg(false).map_err(|err|{
            match err{
                Error::InvalidRegex(mut posi, err_msg) => {
                    println!("{}",exp);
                    while posi > 0{
                        print!(" ");
                        posi -= 1;
                    }
                    println!("^   {}",err_msg);
                    err
                },
                _ => panic!("internal error should not export here")
            }
        })?;

        if comp.code_size >= 0x7fff {
            return Err(Error::InvalidRegex(comp.regposi, "regex too big"));
        }
        comp.program.reserve(comp.code_size);

        comp.regparse = peek_nth(exp.chars());
        comp.regposi = 0;
        comp.paren_count = 1;
        comp.emit_code = true;
        comp.regc(Opcode::BEGIN.into());
        let (_, _) = comp.reg(false)?;
        let regexp = Regexp::new(comp);
        // let scan = PragramCounter(1);
        // TODO: 
        Ok(regexp)
    }
    /*
     *  reg - regular expression, i.e. main body or parenthesized thing
     *  Caller must absorb opening parenthesis.
     */
    fn reg(&mut self, paren: bool) -> Result<(PC, CompStatus), Error> {
        let parno = self.paren_count;
        let mut ret = PC::NULL;
        let mut flags = CompStatus::HASWIDTH;
        if paren {
            if self.paren_count >= MAX_SUB_EXP {
                return Err(Error::InvalidRegex(self.regposi, "too many ()"));
            }
            self.paren_count += 1;
            ret = self.regnode(Opcode::open_n(parno));
        }
        let (br, branch_flags) = self.regbranch()?;
        if paren {
            self.regtail(ret, br)?; /* OPEN -> first. */
        } else {
            ret = br;
        }
        flags &= !(!branch_flags & CompStatus::HASWIDTH); /* if branch is not report has width, clear it */
        flags |= branch_flags & CompStatus::SPSTART;
        loop {
            match self.exp_peak() {
                Some(ch) if ch == '|' => {
                    self.exp_advance();
                    let (br, branch_flags) = self.regbranch()?;
                    self.regtail(ret, br)?; /* BRANCH -> BRANCH. */
                    flags &= !(!branch_flags & CompStatus::HASWIDTH);
                    flags |= branch_flags & CompStatus::SPSTART;
                }
                _ => {
                    break;
                }
            }
        }
        //current exp compile end. hook every thing to end node
        let end = self.regnode(if paren {
            Opcode::close_n(parno)
        } else {
            Opcode::END
        });
        self.regtail(ret, end)?;

        /* Hook the tails of the branches to the closing node. */
        let mut branch = ret;
        self.regoptail(branch, end)?;
        while let Some(next_branch) = self.regnext(branch) {
            self.regoptail(next_branch, end)?;
            branch = next_branch;
        }

        /* Check for proper termination. */
        if paren && self.exp_next() != Some(')') {
            return Err(Error::InvalidRegex(self.regposi,"unterminated ()"));
        } else if !paren && self.exp_peak().is_some() {
            if self.exp_peak() == Some(')') {
                return Err(Error::InvalidRegex(self.regposi,"unmatched ()"));
            } else {
                return Err(Error::InvalidRegex(self.regposi,"internal error: junk on end"));
            }
        }
        Ok((ret, flags))
    }
    /*
    - regbranch - one alternative of an | operator
    *
    * Implements the concatenation operator.
    */
    fn regbranch(&mut self) -> Result<(PC, CompStatus), Error> {
        let mut flags = CompStatus::WORST;
        let ret = self.regnode(Opcode::BRANCH);
        let mut chain = PC::NULL;
        while let Some(ch) = self.exp_peak() {
            if ch == '|' || ch == ')' {
                break;
            }
            let (latest, piece_flags) = self.regpiece()?;
            flags |= piece_flags & CompStatus::HASWIDTH;
            if chain == PC::NULL {
                /* First piece. */
                flags |= piece_flags & CompStatus::SPSTART;
            } else {
                self.regtail(chain, latest)?;
            }
            chain = latest;
        }
        if chain == PC::NULL {
            /* Loop ran zero times. */
            self.regnode(Opcode::NOTHING);
        }

        Ok((ret, flags))
    }
    /*
    - regpiece - something followed by possible [*+?]
    *
    * Note that the branching code sequences used for ? and the general cases
    * of * and + are somewhat optimized:  they use the same NOTHING node as
    * both the endmarker for their branch list and the body of the last branch.
    * It might seem that this node could be dispensed with entirely, but the
    * endmarker role is not redundant.
    */
    fn regpiece(&mut self) -> Result<(PC, CompStatus), Error> {
        let flags;
        let (ret, atom_flags) = self.regatom()?;

        let op = self.exp_peak().unwrap_or('\0');
        if !is_repn(op) {
            flags = atom_flags;
            return Ok((ret, flags));
        }
        if !atom_flags.contains(CompStatus::HASWIDTH) && op != '?' {
            return Err(Error::InvalidRegex(self.regposi,"*+ operand could be empty"));
        }
        flags = match op {
            '*' => CompStatus::WORST | CompStatus::SPSTART,
            '+' => CompStatus::WORST | CompStatus::SPSTART | CompStatus::HASWIDTH,
            '?' => CompStatus::WORST,
            _ => unreachable!(),
        };
        let enough_simple = atom_flags.contains(CompStatus::SIMPLE);
        if op == '*' && enough_simple {
            self.reginsert(Opcode::STAR, ret)?;
        } else if op == '*' {
            self.reginsert(Opcode::BRANCH, ret)?;
            let back = self.regnode(Opcode::BACK);
            self.regoptail(ret, back)?;
            self.regoptail(ret, ret)?;
            let next_branch = self.regnode(Opcode::BRANCH);
            self.regtail(ret, next_branch)?;
            let nothing = self.regnode(Opcode::NOTHING);
            self.regtail(ret, nothing)?;
        } else if op == '+' && enough_simple {
            self.reginsert(Opcode::PLUS, ret)?;
        } else if op == '+' {
            let branch = self.regnode(Opcode::BRANCH);
            self.regtail(ret, branch)?;
            let back = self.regnode(Opcode::BACK);
            self.regtail(back, ret)?;
            let next_branch = self.regnode(Opcode::BRANCH);
            self.regtail(branch, next_branch)?;
            let nothing = self.regnode(Opcode::NOTHING);
            self.regtail(ret, nothing)?;
        } else if op == '?' {
            self.reginsert(Opcode::BRANCH, ret)?;
            let branch = self.regnode(Opcode::BRANCH);
            self.regtail(ret, branch)?;
            let nothing = self.regnode(Opcode::NOTHING);
            self.regtail(ret, nothing)?;
            self.regoptail(ret, nothing)?;
        }
        self.exp_advance();
        if let Some(ch) = self.exp_peak() {
            if is_repn(ch) {
                return Err(Error::InvalidRegex(self.regposi,"nested *?+"));
            }
        }
        Ok((ret, flags))
    }
    /*
    - regatom - the lowest level
    *
    * Optimization:  gobbles an entire sequence of ordinary characters so that
    * it can turn them into a single node, which is smaller to store and
    * faster to run.  Backslashed characters are exceptions, each becoming a
    * separate node; the code is simpler that way and it's not worth fixing.
    */
    fn regatom(&mut self) -> Result<(PC, CompStatus), Error> {
        let mut flags = CompStatus::WORST;
        let ch = self.exp_next();
        let ret;
        match ch {
            Some('^') => {
                ret = self.regnode(Opcode::BOL);
            }
            Some('$') => {
                ret = self.regnode(Opcode::EOL);
            }
            Some('.') => {
                ret = self.regnode(Opcode::ANY);
                flags |= CompStatus::HASWIDTH | CompStatus::SIMPLE;
            }
            Some('[') => {
                todo!()
            }

            Some('(') => {
                let (sub_ret, sub_reg_flags) = self.reg(true)?;
                ret = sub_ret;
                flags |= sub_reg_flags & (CompStatus::HASWIDTH | CompStatus::SPSTART);
            }
            Some('|') | Some(')') | None => {
                return Err(Error::InvalidRegex(self.regposi,"internal error: \0|) unexpected"))
            }
            Some('?') | Some('+') | Some('*') => {
                return Err(Error::InvalidRegex(self.regposi,"?+* follows nothing"));
            }
            Some('\\') => {
                if let Some(ch) = self.exp_peak() {
                    self.exp_advance();
                    ret = self.regnode(Opcode::EXACTLY);
                    self.regchar(ch);
                    flags |= CompStatus::HASWIDTH | CompStatus::SIMPLE;
                } else {
                    return Err(Error::InvalidRegex(self.regposi,"trailing \\"));
                }
            }

            // nomal case, single character or a string of characters handle here.
            Some(ch) => {
                ret = self.regnode(Opcode::EXACTLY);
                if is_mata(ch) {
                    return Err(Error::InvalidRegex(self.regposi,"internal error: mata char unexpected"));
                }
                // we have consume a ch. append it to EXACTLY oprand.
                self.regchar(ch);
                let mut len = 0;
                // absorb as many as simple char
                while let Some(ch) = self.exp_peak_nth(len) {
                    if is_mata(ch) {
                        break;
                    }
                    len += 1;
                }
                // consider "abcd*", we must back one character, since '*' is left combination
                if let Some(first_mata) = self.exp_peak_nth(len) {
                    if len > 0 && is_repn(first_mata) {
                        len -= 1; /* Back off clear of ?+* operand. */
                    }
                }
                flags |= CompStatus::HASWIDTH;
                // we have append first char
                if len == 0 {
                    flags |= CompStatus::SIMPLE;
                }

                for _ in 0..len {
                    let ch = self.exp_next().expect("this fine");
                    self.regchar(ch);
                }
                // chars boundary
                self.regchar('\0');
            }
        };
        Ok((ret, flags))
    }

    // emit a node at tail of program, this shouldn't fail
    fn regnode(&mut self, op: Opcode) -> PC {
        if self.emit_code {
            self.program.emit_node(op)
        } else {
            self.code_size += 3;
            PragramCounter(0)
        }
    }
    //  emit (if appropriate) a byte of code at tail of program. shouldn't fail
    fn regc(&mut self, byte: u8) {
        if self.emit_code {
            self.program.emit_byte(byte);
        } else {
            self.code_size += 1;
        }
    }
    // emit a utf-8 char at tail of program
    fn regchar(&mut self, ch: char) {
        if !self.emit_code {
            self.code_size += 4;
            return;
        }
        let ch = ch as u32;
        let buf = ch.to_be_bytes();
        for byte in buf {
            self.program.emit_byte(byte);
        }
    }

    // insert an operator in front of already-emitted operand
    fn reginsert(&mut self, op: Opcode, opnd: PC) -> Result<(), Error> {
        if self.emit_code {
            self.program.insert_node(op, opnd)
        } else {
            self.code_size += 3;
            Ok(())
        }
    }

    fn regnext(&self, pc: PC) -> Option<PC> {
        self.program.next_at(pc)
    }

    fn regtail(&mut self, chain: PC, new_tail: PC) -> Result<(), Error> {
        if self.emit_code {
            self.program.update_tail(chain, new_tail)
        } else {
            Ok(())
        }
    }
    fn regoptail(&mut self, branch: PC, new_tail: PC) -> Result<(), Error> {
        if self.emit_code {
            self.program.update_oprand_tail(branch, new_tail)
        } else {
            Ok(())
        }
    }
    fn exp_peak(&mut self) -> Option<char> {
        self.regparse.peek().copied()
    }
    fn exp_peak_nth(&mut self, n: usize) -> Option<char> {
        self.regparse.peek_nth(n).copied()
    }
    fn exp_advance(&mut self) {
        self.regposi += 1;
        self.regparse.next();
    }
    fn exp_next(&mut self) -> Option<char> {
        self.regposi += 1;
        self.regparse.next()
    }
}


/*
 * Work-variable struct for vm execute.
 */
struct Exec<'a>{
    input: &'a str,
}

impl<'a> Exec<'a>{
    
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_program_out_of_bound() {
        let mut prog = Program::new();
        assert_eq!(prog.pc, PragramCounter(0));

        let old_pc = prog.emit_node(Opcode::ANY);
        assert_eq!(prog.is_node_out_of_bound(old_pc), Ok(()));
        assert_eq!(
            prog.is_node_out_of_bound(old_pc + 1),
            Err(Error::OutOfBound)
        );
        assert_eq!(
            prog.is_node_out_of_bound(old_pc + 2),
            Err(Error::OutOfBound)
        );
    }

    #[test]
    fn test_program_emit_byte() {
        let mut prog = Program::new();
        prog.emit_byte(10);
        prog.emit_byte(12);
        prog.emit_byte(100);
        assert_eq!(prog.bincode, vec![10, 12, 100]);
        assert_eq!(prog.pc, PragramCounter(3));
    }
    #[test]
    fn test_program_emit_node() {
        let mut prog = Program::new();
        {
            let pc = prog.emit_node(Opcode::BEGIN);
            assert_eq!(prog.bincode, vec![Opcode::BEGIN.into(), 0, 0]);
            assert_eq!(pc, PragramCounter(0));
            assert_eq!(Ok(Opcode::BEGIN), prog.opcode_at(pc));
        }
        {
            let pc = prog.emit_node(Opcode::ANY);
            assert_eq!(
                prog.bincode,
                vec![Opcode::BEGIN.into(), 0, 0, Opcode::ANY.into(), 0, 0]
            );
            assert_eq!(pc, PragramCounter(3));
            assert_eq!(Ok(Opcode::ANY), prog.opcode_at(pc));
        }
    }
    #[test]
    fn test_program_updata_tail() {
        let mut proj = Program::new();
        let first = proj.emit_node(Opcode::BEGIN);
        let second = proj.emit_node(Opcode::ANY);
        // first => second
        assert!(proj.update_tail(first, second).is_ok());
        assert_eq!(proj.next_at(first), Some(second));
        assert_eq!(proj.next_at(second), None);
    }
    #[test]
    fn test_program_updata_tail_back() {
        let mut proj = Program::new();
        let first = proj.emit_node(Opcode::BRANCH);
        let second = proj.emit_node(Opcode::BACK);
        assert!(proj.update_tail(second, first).is_ok());
        assert_eq!(proj.next_at(second), Some(first));
        assert_eq!(proj.next_at(first), None);
    }
    #[test]
    fn test_program_insert_node() {
        let mut prog = Program::new();

        let f = prog.emit_node(Opcode::ANY);
        prog.insert_node(Opcode::STAR, f).unwrap();
        assert_eq!(
            prog.bincode,
            vec![Opcode::STAR.into(), 0, 0, Opcode::ANY.into(), 0, 0]
        );

        let s = prog.emit_node(Opcode::ANY);
        prog.insert_node(Opcode::PLUS, s).unwrap();
        assert_eq!(
            prog.bincode,
            vec![
                Opcode::STAR.into(),
                0,
                0,
                Opcode::ANY.into(),
                0,
                0,
                Opcode::PLUS.into(),
                0,
                0,
                Opcode::ANY.into(),
                0,
                0
            ]
        )
    }

    fn regcomp(exp: &str)->Result<(), Error>{
        Comp::regcomp(exp).map(|_| ())
    }
    
    #[test]
    fn test_regex_comp_ok_simple() {
        assert!(regcomp("a").is_ok());
        assert!(regcomp("abcdefg").is_ok());
        assert!(regcomp("a|b").is_ok());
        assert!(regcomp("^a").is_ok());
        assert!(regcomp("a$").is_ok());
        assert!(regcomp("a*").is_ok());
        assert!(regcomp("a?").is_ok());
        assert!(regcomp("a+").is_ok());
        assert!(regcomp("(a|b)c*d+").is_ok());
        assert!(regcomp("(a+|b)?").is_ok());
        assert!(regcomp("hello world! Have a good day.").is_ok());
        
    }
    #[test]
    fn test_regex_comp_err_simple(){
        assert!(regcomp("a**").is_err())
    }

    #[test]
    fn test_regex_program_display(){
        let re = Comp::regcomp("(a)+").unwrap();
        println!("{}", re.program);
    }
}
