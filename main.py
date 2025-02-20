from parsers.site1_parser import Site1Parser
from parsers.site2_parser import Site2Parser
from parsers.site3_parser import Site3Parser

if __name__ == "__main__":
    # Парсинг отзывов с каждого сайта
    site1_parser = Site1Parser()
    site1_reviews = site1_parser.parse()

    site2_parser = Site2Parser()
    site2_reviews = site2_parser.parse()

    site3_parser = Site3Parser() 
    site3_reviews = site3_parser.parse()