from antlr4 import InputStream, CommonTokenStream

# Import the generated lexer and parser
from cfg_grammar.NamespaceLexer import NamespaceLexer
from cfg_grammar.NamespaceParser import NamespaceParser

def parse_config(input_text):
    # Create an input stream from the input text
    input_stream = InputStream(input_text)

    # Create a lexer
    lexer = NamespaceLexer(input_stream)

    # Create a token stream from the lexer
    token_stream = CommonTokenStream(lexer)

    # Create a parser from the token stream
    parser = NamespaceParser(token_stream)

    # Parse the input and get the parse tree
    tree = parser.namespace()

    extracted_dict = {}

    pairs = tree.pairs().pair()
    for pair in pairs:
        key = pair.ID().getText()
        value = pair.value()
        if value.INT() is not None:
            dict_value = int(value.INT().getText())
        elif value.FLOAT() is not None:
            dict_value = float(value.FLOAT().getText())
        elif value.BOOL() is not None:
            dict_value = bool(value.BOOL().getText())
        elif value.STRING() is not None:
            dict_value = str(value.STRING().getText())[1:-1]
        else:
            raise ValueError('type did not recognized')
    
        extracted_dict[key] = dict_value
    
    return extracted_dict

if __name__ == "__main__":
    parse_config("Namespace(sh_degree=3, source_path='/home/mbortolon/data/datasets/360_v2/bicycle', model_path='./output/ec0d365d-5', images='images', resolution=-1, white_background=False, data_device='cuda', eval=True)")