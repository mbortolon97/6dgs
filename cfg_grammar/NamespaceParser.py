# Generated from Namespace.g4 by ANTLR 4.9.3
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys

def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\f")
        buf.write("\35\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\3\2\3\2\3\2\3\2\3")
        buf.write("\3\3\3\3\3\7\3\22\n\3\f\3\16\3\25\13\3\3\4\3\4\3\4\3\4")
        buf.write("\3\5\3\5\3\5\2\2\6\2\4\6\b\2\3\3\2\7\n\2\31\2\n\3\2\2")
        buf.write("\2\4\16\3\2\2\2\6\26\3\2\2\2\b\32\3\2\2\2\n\13\7\3\2\2")
        buf.write("\13\f\5\4\3\2\f\r\7\4\2\2\r\3\3\2\2\2\16\23\5\6\4\2\17")
        buf.write("\20\7\5\2\2\20\22\5\6\4\2\21\17\3\2\2\2\22\25\3\2\2\2")
        buf.write("\23\21\3\2\2\2\23\24\3\2\2\2\24\5\3\2\2\2\25\23\3\2\2")
        buf.write("\2\26\27\7\13\2\2\27\30\7\6\2\2\30\31\5\b\5\2\31\7\3\2")
        buf.write("\2\2\32\33\t\2\2\2\33\t\3\2\2\2\3\23")
        return buf.getvalue()


class NamespaceParser ( Parser ):

    grammarFileName = "Namespace.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'Namespace('", "')'", "','", "'='" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "BOOL", "INT", "FLOAT", "STRING", "ID", 
                      "WS" ]

    RULE_namespace = 0
    RULE_pairs = 1
    RULE_pair = 2
    RULE_value = 3

    ruleNames =  [ "namespace", "pairs", "pair", "value" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    BOOL=5
    INT=6
    FLOAT=7
    STRING=8
    ID=9
    WS=10

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.9.3")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None



    class NamespaceContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def pairs(self):
            return self.getTypedRuleContext(NamespaceParser.PairsContext,0)


        def getRuleIndex(self):
            return NamespaceParser.RULE_namespace

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNamespace" ):
                listener.enterNamespace(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNamespace" ):
                listener.exitNamespace(self)




    def namespace(self):

        localctx = NamespaceParser.NamespaceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_namespace)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 8
            self.match(NamespaceParser.T__0)
            self.state = 9
            self.pairs()
            self.state = 10
            self.match(NamespaceParser.T__1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PairsContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def pair(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(NamespaceParser.PairContext)
            else:
                return self.getTypedRuleContext(NamespaceParser.PairContext,i)


        def getRuleIndex(self):
            return NamespaceParser.RULE_pairs

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPairs" ):
                listener.enterPairs(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPairs" ):
                listener.exitPairs(self)




    def pairs(self):

        localctx = NamespaceParser.PairsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_pairs)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 12
            self.pair()
            self.state = 17
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==NamespaceParser.T__2:
                self.state = 13
                self.match(NamespaceParser.T__2)
                self.state = 14
                self.pair()
                self.state = 19
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PairContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.key = None # Token

        def value(self):
            return self.getTypedRuleContext(NamespaceParser.ValueContext,0)


        def ID(self):
            return self.getToken(NamespaceParser.ID, 0)

        def getRuleIndex(self):
            return NamespaceParser.RULE_pair

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPair" ):
                listener.enterPair(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPair" ):
                listener.exitPair(self)




    def pair(self):

        localctx = NamespaceParser.PairContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_pair)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 20
            localctx.key = self.match(NamespaceParser.ID)
            self.state = 21
            self.match(NamespaceParser.T__3)
            self.state = 22
            self.value()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ValueContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self):
            return self.getToken(NamespaceParser.INT, 0)

        def FLOAT(self):
            return self.getToken(NamespaceParser.FLOAT, 0)

        def BOOL(self):
            return self.getToken(NamespaceParser.BOOL, 0)

        def STRING(self):
            return self.getToken(NamespaceParser.STRING, 0)

        def getRuleIndex(self):
            return NamespaceParser.RULE_value

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterValue" ):
                listener.enterValue(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitValue" ):
                listener.exitValue(self)




    def value(self):

        localctx = NamespaceParser.ValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_value)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 24
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << NamespaceParser.BOOL) | (1 << NamespaceParser.INT) | (1 << NamespaceParser.FLOAT) | (1 << NamespaceParser.STRING))) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





