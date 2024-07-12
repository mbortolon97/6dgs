// Namespace.g4

grammar Namespace;

namespace: 'Namespace(' pairs ')' ;

pairs: pair (',' pair)* ;

pair: key=ID '=' value ;

value: INT
    | FLOAT
    | BOOL
    | STRING
    ;

BOOL: 'True' | 'False' | 'true' | 'false';
INT:   [+-]?[0-9]+ ;
FLOAT: [+-]?[0-9]+ '.' [0-9]* ;
STRING: ['"](~('\'' | '\\' | '\n' | '\r'))*?['"] ;
ID: [a-zA-Z_][a-zA-Z0-9_]* ;

WS: [ \t\r\n]+ -> skip ;

// fragment ESC : ('\\''|'\\"'|'\\\\');
