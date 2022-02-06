#![allow(clippy::upper_case_acronyms)]

use bitflags::bitflags;
use itertools::{peek_nth, PeekNth};
use num_enum::{IntoPrimitive, TryFromPrimitive};

use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, AddAssign, Sub},
    rc::Rc,
    str::Chars,
};

const MAX_SUB_EXP: usize = 9;
const META_CHAR: &str = "^$.[()|?+*\\";
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    OutOfBound,
    InvalidOpcode,
    InvaildOperand,
    InvalidRegex(usize, &'static str),
    ExecuteFailed(usize, &'static str),
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
    bincode: Rc<RefCell<Vec<u8>>>,
}
impl Program {
    pub fn new() -> Program {
        Program {
            pc: PragramCounter(0),
            bincode: Rc::new(RefCell::new(vec![])),
        }
    }
    pub fn clone_with_pc(&self, pc: PC) -> Program {
        Program {
            pc,
            bincode: Rc::clone(&self.bincode),
        }
    }
    pub fn reserve(&mut self, additional: usize) {
        self.bincode.borrow_mut().reserve(additional);
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
        Opcode::try_from(self.bincode.borrow()[pc.0]).map_err(|_| Error::InvalidOpcode)
    }
    // return the next pc at the position pc in bincode.
    pub fn next_at(&self, pc: PC) -> Option<PC> {
        self.is_node_out_of_bound(pc).ok()?;
        let high_byte = self.bincode.borrow()[(pc + 1).0] as usize;
        let low_byte = self.bincode.borrow()[(pc + 2).0] as usize;
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
        self.bincode.borrow_mut().push(opcode.into());
        self.bincode.borrow_mut().push(0);
        self.bincode.borrow_mut().push(0);
        self.pc += 3;
        old_pc
    }
    pub fn emit_byte(&mut self, byte: u8) {
        self.bincode.borrow_mut().push(byte);
        self.pc += 1;
    }

    // insert an operator in front of already-emitted operand
    // Means relocating the operand.
    pub fn insert_node(&mut self, opcode: Opcode, opnd: PC) -> Result<(), Error> {
        self.is_node_out_of_bound(opnd)?;
        self.bincode.borrow_mut().push(0);
        self.bincode.borrow_mut().push(0);
        self.bincode.borrow_mut().push(0);
        let begin = opnd.0;
        let end = self.bincode.borrow().len();
        for i in (begin + 3..end).rev() {
            let tmp = self.bincode.borrow()[i - 3];
            self.bincode.borrow_mut()[i] = tmp;
        }

        self.bincode.borrow_mut()[begin] = opcode.into();
        self.bincode.borrow_mut()[begin + 1] = 0;
        self.bincode.borrow_mut()[begin + 2] = 0;
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
        self.bincode.borrow_mut()[chain.0 + 1] = ((offset >> 8) & 0o177) as u8;
        self.bincode.borrow_mut()[chain.0 + 2] = (offset & 0o377) as u8;
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

    fn program(&self) -> Vec<u8> {
        self.bincode.borrow().to_owned()
    }
    // return char(four byte) at pc
    fn char_at(&self, pc: PC) -> Option<char> {
        let pc = pc.0;
        let bincode = self.bincode.borrow();
        if pc + 4 > bincode.len() {
            None
        } else {
            let mut buf = [0; 4];
            buf.copy_from_slice(&bincode[pc..pc + 4]);
            let ch = u32::from_be_bytes(buf);
            char::from_u32(ch)
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
                if opcode == Opcode::BACK {
                    write!(f, "({})", pc.0 - (pc.0 - next.0))?;
                } else {
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
                    digit.copy_from_slice(&self.bincode.borrow_mut()[base..base + 4]);
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
    pub fn program(self) -> Program {
        self.program
    }

    /*
    -   regcomp  compile a regular expression into internal code
    *
    *   compile regex will be two phase process. first for verify the validity of the regex, don't emit code.
    *
    */
    pub fn regcomp(exp: &str) -> Result<Comp, Error> {
        let mut comp = Comp {
            regparse: peek_nth(exp.chars()),
            regposi: 0,
            paren_count: 1,
            program: Program::new(),
            emit_code: false,
            code_size: 0,
        };
        comp.regc(Opcode::BEGIN.into());
        comp.reg(false).map_err(|err| match err {
            Error::InvalidRegex(mut posi, err_msg) => {
                println!("{}", exp);
                while posi > 0 {
                    print!(" ");
                    posi -= 1;
                }
                println!("^   {}", err_msg);
                err
            }
            _ => panic!("internal error should not export here"),
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
        // let regexp = Regexp::new(comp);
        // let scan = PragramCounter(1);
        // TODO:
        Ok(comp)
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
            return Err(Error::InvalidRegex(self.regposi, "unterminated ()"));
        } else if !paren && self.exp_peak().is_some() {
            if self.exp_peak() == Some(')') {
                return Err(Error::InvalidRegex(self.regposi, "unmatched ()"));
            } else {
                return Err(Error::InvalidRegex(
                    self.regposi,
                    "internal error: junk on end",
                ));
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
            return Err(Error::InvalidRegex(
                self.regposi,
                "*+ operand could be empty",
            ));
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
            /*
                EXAMPLE
                (ret)*remain:

                  +---------------------------+
                  |                           |
                  v                           |
                BRANCH   --->   ret   --->   BACK   BRANCH   --->   NOTHING   --->   remain
                  |                                  ^  |                               ^
                  |                                  |  v                               |
                  +----------------------------------+  +-------------------------------+
            */
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
            /*
                EXAMPLE
                (ret)+remain:

                   +-----------------------------+
                   |                             |
                   v                             | 
                  ret   --->   BRANCH   --->   BACK    BRANCH   --->   NOTHING   --->   remain
                                  |                     ^  |                               ^
                                  |                     |  v                               |
                                  +----------------------  --------------------------------+
            */
            let branch = self.regnode(Opcode::BRANCH);
            self.regtail(ret, branch)?;
            let back = self.regnode(Opcode::BACK);
            self.regtail(back, ret)?;
            let next_branch = self.regnode(Opcode::BRANCH);
            self.regtail(branch, next_branch)?;
            let nothing = self.regnode(Opcode::NOTHING);
            self.regtail(ret, nothing)?;
        } else if op == '?' {
            /*
                EXAMPLE
                (ret)?remain

                 
                BRANCH   --->   ret   --->   BRANCH   --->   NOTHING   --->   remain
                  |                           ^  v                               ^
                  +---------------------------+  +-------------------------------+

             */
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
                return Err(Error::InvalidRegex(self.regposi, "nested *?+"));
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
                // TODO:
                todo!()
            }

            Some('(') => {
                let (sub_ret, sub_reg_flags) = self.reg(true)?;
                ret = sub_ret;
                flags |= sub_reg_flags & (CompStatus::HASWIDTH | CompStatus::SPSTART);
            }
            Some('|') | Some(')') | None => {
                return Err(Error::InvalidRegex(
                    self.regposi,
                    "internal error: \0|) unexpected",
                ))
            }
            Some('?') | Some('+') | Some('*') => {
                return Err(Error::InvalidRegex(self.regposi, "?+* follows nothing"));
            }
            Some('\\') => {
                if let Some(ch) = self.exp_peak() {
                    self.exp_advance();
                    ret = self.regnode(Opcode::EXACTLY);
                    self.regchar(ch);
                    flags |= CompStatus::HASWIDTH | CompStatus::SIMPLE;
                } else {
                    return Err(Error::InvalidRegex(self.regposi, "trailing \\"));
                }
            }

            // nomal case, single character or a string of characters handle here.
            Some(ch) => {
                ret = self.regnode(Opcode::EXACTLY);
                if is_mata(ch) {
                    return Err(Error::InvalidRegex(
                        self.regposi,
                        "internal error: mata char unexpected",
                    ));
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

#[derive(Clone, Default)]
struct SubMatch<'a> {
    start: Option<Chars<'a>>,
    end: Option<Chars<'a>>,
}

impl<'a> SubMatch<'a> {
    fn new(start: Chars<'a>, end: Chars<'a>) -> SubMatch<'a> {
        SubMatch {
            start: Some(start),
            end: Some(end),
        }
    }
    fn start_seted(&self) -> bool {
        self.start.is_some()
    }
    fn end_seted(&self) -> bool {
        self.end.is_some()
    }
    fn set_start(&mut self, start: Chars<'a>) {
        self.start = Some(start);
    }
    fn set_end(&mut self, end: Chars<'a>) {
        self.end = Some(end);
    }
    fn is_fine(&self) -> bool {
        self.start.is_some() && self.end.is_some()
    }
}
/*
 * Work-variable struct for vm execute.
 */
struct Exec<'a> {
    reginput: Chars<'a>,
    regbol: Chars<'a>,
    looked_len: usize,
    submatchs: [SubMatch<'a>; MAX_SUB_EXP],
}

impl<'a> Exec<'a> {
    pub fn regexec(comp: &mut Comp, s: &'a str) -> Result<Exec<'a>, Error> {
        let mut exec = Exec {
            reginput: s.chars(),
            regbol: s.chars(),
            looked_len: 0,
            submatchs: Default::default(),
        };
        let prog = &mut comp.program;
        let pc = PragramCounter(0);
        let first_op = prog
            .opcode_at(pc)
            .map_err(|_| Error::ExecuteFailed(0, "corrupted regex program"))?;
        // program valid ?
        if first_op != Opcode::BEGIN {
            return Err(Error::ExecuteFailed(0, "corruptd regex program"));
        }
        let mut test_point = s.chars();
        // ignore first BEGIN
        prog.pc += 1;
        exec.regtry(prog)?;
        Ok(exec)
    }

    fn regtry(&mut self, prog: &mut Program) -> Result<(), Error> {
        let match_begin = self.reginput.clone();

        if self.regmatch(prog).is_ok() {
            let matched = SubMatch::new(match_begin, self.reginput.clone());
            self.submatchs[0] = matched;
        }
        Ok(())
    }

    /*
    - regmatch - main matching routine
    *
    * Conceptually the strategy is simple:  check to see whether the current
    * node matches, call self recursively to see whether the rest matches,
    * and then act accordingly.  In practice we make some effort to avoid
    * recursion, in particular by going through "ordinary" nodes (that don't
    * need to know whether the rest of the match failed) by a loop instead of
    * by recursion.
    */
    fn regmatch(&mut self, prog: &mut Program) -> Result<(), Error> {
        let mut scan = prog.pc;

        while scan != PC::NULL {
            let mut next = prog.next_at(scan).unwrap_or(PC::NULL);
            let opcode = prog
                .opcode_at(scan)
                .map_err(|_| Error::ExecuteFailed(self.looked_len, "regex program corrupted"))?;
            match opcode {
                Opcode::BOL => {
                    if self.reginput.as_str() != self.regbol.as_str() {
                        return Err(Error::ExecuteFailed(
                            self.looked_len,
                            "match begin of line failed",
                        ));
                    }
                }
                Opcode::EOL => {
                    if self.next_char() != None {
                        return Err(Error::ExecuteFailed(
                            self.looked_len,
                            "match end of line failed",
                        ));
                    }
                }
                Opcode::ANY => {
                    if self.next_char() == None {
                        return Err(Error::ExecuteFailed(
                            self.looked_len,
                            "match any char failed",
                        ));
                    }
                }
                Opcode::EXACTLY => {
                    let opnd = prog.operand_at(scan).ok_or(Error::InvaildOperand)?;
                    let mut idx = opnd.0;
                    let mut c;
                    loop {
                        let mut tmp_ary = [0; 4];
                        tmp_ary.copy_from_slice(&prog.bincode.borrow()[idx..idx + 4]);
                        c = u32::from_be_bytes(tmp_ary);
                        if c == 0 {
                            break;
                        }
                        match self.next_char() {
                            Some(ch) if ch as u32 == c => idx += 4,
                            _ => {
                                return Err(Error::ExecuteFailed(
                                    self.looked_len,
                                    "match exactly failed",
                                ))
                            }
                        }
                    }
                }
                Opcode::ANYOF => {}
                Opcode::ANYBUT => {}
                Opcode::NOTHING => { /* do nothing( */ }
                Opcode::BACK => { /* do nothing( */ }

                Opcode::OPEN1
                | Opcode::OPEN2
                | Opcode::OPEN3
                | Opcode::OPEN4
                | Opcode::OPEN5
                | Opcode::OPEN6
                | Opcode::OPEN7
                | Opcode::OPEN8
                | Opcode::OPEN9 => {
                    let no = prog.opcode_at(scan).unwrap() as u8 - Opcode::OPEN as u8;
                    let no = no as usize;
                    let input = self.reginput.clone();
                    let mut split = prog.clone_with_pc(next);
                    if self.regmatch(&mut split).is_ok() {
                        if !self.submatchs[no].start_seted() {
                            self.submatchs[no].set_start(input);
                        }
                    } else {
                        return Err(Error::ExecuteFailed(self.looked_len, "submatch failed"));
                    }
                }

                Opcode::CLOSE1
                | Opcode::CLOSE2
                | Opcode::CLOSE3
                | Opcode::CLOSE4
                | Opcode::CLOSE5
                | Opcode::CLOSE6
                | Opcode::CLOSE7
                | Opcode::CLOSE8
                | Opcode::CLOSE9 => {
                    let no = prog.opcode_at(scan).unwrap() as u8 - Opcode::CLOSE as u8;
                    let no = no as usize;
                    let input = self.reginput.clone();
                    let mut split = prog.clone_with_pc(next);
                    if self.regmatch(&mut split).is_ok() {
                        if !self.submatchs[no].end_seted() {
                            self.submatchs[no].set_end(input);
                        }
                    } else {
                        return Err(Error::ExecuteFailed(self.looked_len, "submatch failed"));
                    }
                }
                Opcode::BRANCH => {
                    let save = self.reginput.clone();

                    match prog.opcode_at(next) {
                        //
                        Ok(op) if op == Opcode::BRANCH => loop {
                            match prog.opcode_at(scan) {
                                Ok(op) if op == Opcode::BRANCH => {
                                    let alternative
                                     = prog.operand_at(scan).unwrap();
                                    let mut prog_clone = prog.clone_with_pc(alternative);
                                    if self.regmatch(&mut prog_clone).is_ok() {
                                        return Ok(());
                                    }
                                    self.reginput = save.clone();
                                    scan = prog.next_at(scan).unwrap_or(PC::NULL);
                                }
                                _ => {
                                    // TODO: 
                                    todo!()
                                }
                            }
                        },
                        // no choice
                        Ok(_) => {
                            next = prog.operand_at(scan).unwrap();
                        }
                        Err(_) => {
                            return Err(Error::ExecuteFailed(
                                self.looked_len,
                                "match branch failed, opcode of next invalid",
                            ));
                        }
                    }
                }
                Opcode::STAR | Opcode::PLUS => {
                    // TODO:
                }
                Opcode::END => return Ok(()),
                _ => {
                    return Err(Error::ExecuteFailed(self.looked_len, "unexpected program"));
                }
            };
            scan = next;
        }

        Err(Error::ExecuteFailed(
            self.looked_len,
            "unexpeced return point",
        ))
    }
    // report how many times something simple would match
    fn regrepeat(&self, prog: &Program, pc: PC) -> usize {
        match prog.opcode_at(pc) {
            Ok(op) if op == Opcode::STAR || op == Opcode::PLUS => { /* ok */ }
            _ => panic!("call repeat with invalid opcode"),
        }
        let node = prog.operand_at(pc).unwrap();
        match prog.opcode_at(node) {
            Ok(Opcode::ANY) => {
                // consider ".*"
                self.reginput.as_str().len()
            }
            Ok(Opcode::ANYOF) => 0,
            Ok(Opcode::ANYBUT) => 0,
            Ok(Opcode::EXACTLY) => {
                // exactly single char
                let mut count = 0;
                let ch = prog
                    .operand_at(node)
                    .expect("EXACTLY follow at least one character ");
                let ch = prog.char_at(ch).expect("regex program corrupted");
                let mut point = self.reginput.clone();
                loop {
                    match point.next() {
                        Some(c) if c == ch => count += 1,
                        _ => break,
                    }
                }
                count
            }
            _ => {
                panic!("call repeat with invalid oprand");
            }
        }
    }
    fn next_char(&mut self) -> Option<char> {
        self.looked_len += 1;
        self.reginput.next()
    }
}

pub struct Regex<'a> {
    program: Program,
    submatchs: [Option<SubMatch<'a>>; MAX_SUB_EXP],
}
impl<'a> Regex<'a> {
    pub fn new(re: &str) -> Result<Regex, Error> {
        todo!()
    }
    pub fn find(&mut self, s: &str) {}
    pub fn captures(&self) {}
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
    fn test_program_char_at() {
        let mut prog = Program::new();
        let mut pc = PragramCounter(0);

        let emit_char = |prog: &mut Program, ch: char| {
            let buf = (ch as u32).to_be_bytes();
            for b in buf {
                prog.emit_byte(b);
            }
        };

        {
            emit_char(&mut prog, 'c');
            assert_eq!(prog.char_at(pc), Some('c'));
            pc += 4;
        }
        {
            emit_char(&mut prog, 'a');
            assert_eq!(prog.char_at(pc), Some('a'));
            pc += 4;
        }
        {
            assert_eq!(prog.char_at(pc), None);
        }
    }
    #[test]
    fn test_program_emit_byte() {
        let mut prog = Program::new();
        prog.emit_byte(10);
        prog.emit_byte(12);
        prog.emit_byte(100);
        assert_eq!(prog.pc, PragramCounter(3));
        assert_eq!(prog.program(), vec![10, 12, 100]);
    }
    #[test]
    fn test_program_emit_node() {
        let mut prog = Program::new();
        {
            let pc = prog.emit_node(Opcode::BEGIN);
            assert_eq!(pc, PragramCounter(0));
            assert_eq!(Ok(Opcode::BEGIN), prog.opcode_at(pc));
            assert_eq!(prog.program(), vec![Opcode::BEGIN.into(), 0, 0]);
        }
        {
            let pc = prog.emit_node(Opcode::ANY);
            assert_eq!(pc, PragramCounter(3));
            assert_eq!(Ok(Opcode::ANY), prog.opcode_at(pc));
            assert_eq!(
                prog.program(),
                vec![Opcode::BEGIN.into(), 0, 0, Opcode::ANY.into(), 0, 0]
            );
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
            prog.program(),
            vec![Opcode::STAR.into(), 0, 0, Opcode::ANY.into(), 0, 0]
        );

        let s = prog.emit_node(Opcode::ANY);
        prog.insert_node(Opcode::PLUS, s).unwrap();
        assert_eq!(
            prog.program(),
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

    fn regcomp(exp: &str) -> Result<(), Error> {
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
    fn test_regex_comp_err_simple() {
        assert!(regcomp("a**").is_err())
    }

    #[test]
    fn test_regex_program_display() {
        let re = Comp::regcomp("a*").unwrap();
        println!("{}", re.program);
    }

    fn regexec(re: &str, s: &str) -> Result<(), Error> {
        let mut comp = Comp::regcomp(re)?;
        Exec::regexec(&mut comp, s).map(|_| ())
    }
    #[test]
    fn test_regex_exec_ok_simple() {
        assert!(regexec("a", "a").is_ok());
        assert!(regexec("abcedf", "abcedf").is_ok());
        assert!(regexec("a|b|c", "a").is_ok());
        assert!(regexec("a|b|c", "c").is_ok());
        assert!(regexec("a|b|.", "d").is_ok());
        assert!(regexec("(a)", "a").is_ok());
        assert!(regexec("((((((a))))))", "a").is_ok());
        assert!(regexec("a*", "aaaaaaaaaa").is_ok());
        assert!(regexec("a*|b", "").is_ok());
        assert!(regexec("a*b+c?", "aaaab").is_ok());
        assert!(regexec("(a)|(b)", "a").is_ok());
        assert!(regexec("(a)*", "").is_ok());
        assert!(regexec("(a)*", "aaa").is_ok());
    }
    #[test]
    fn test_regex_exec_err_simple() {
        assert!(regexec("a*", "bbb").is_err());
    }
}
