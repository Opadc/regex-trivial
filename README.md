Henry Spencer's [regexp.old](https://github.com/garyhouston/regexp.old) lib rewrite to rust, for fun.  

I found Spencer's lib from excellent blog [Implementing Regular Expressions](https://swtch.com/~rsc/regexp/)  of Russ Cox and Zhihu article [如何从零写一个正则表达式引擎？](https://www.zhihu.com/question/27434493/answer/36803124).   


### regular expression syntnx  

       A regular expression is zero or more branches, separated by `|'.  It matches anything that matches one of the branches.

       A branch is zero or more pieces, concatenated.  It matches a match for the first, followed by a match for the second, etc.

       A  piece is an atom possibly followed by `*', `+', or `?'.  An atom followed by `*' matches a sequence of 0 or more matches of the atom.  An atom followed by `+' matches a sequence of 1 or more matches of the atom.  An atom followed by `?' matches a match of the atom, or the null string.

       An atom is a regular expression in parentheses (matching a match for the regular expression), a range (see below), `.'  (matching any  single  charac‐ter),  `^'  (matching the null string at the beginning of the input string), `$' (matching the null string at the end of the input string), a `\' followed by a single character (matching that character), or a single character with no other significance (matching that character).

       A range is a sequence of characters enclosed in `[]'.  It normally matches any single character from the sequence.  If the sequence begins  with  `^', it  matches any single character not from the rest of the sequence.  If two characters in the sequence are separated by `-', this is shorthand for the full list of ASCII characters between them (e.g. `[0-9]' matches any decimal digit).  To include a literal `]' in the  sequence,  make  it  the  first character (following a possible `^').  To include a literal `-', make it the first or last character.   

### tests
       
### warning
Nothing new here, this is almost directly translate C to Rust, and I will try my best to optimize



