import re
from typing import Iterable

from docutils.parsers.rst import Parser as RstParser

REGEX = re.compile(r".*#.*doc\s*:\s*hide")


def remove_lines_with_hide(inputstring):
    return "\n".join(line for line in inputstring.splitlines() if not REGEX.match(line))


class Parser(RstParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse(self, inputstring, document):
        inputstring = remove_lines_with_hide(inputstring)
        print(inputstring)
        return super().parse(inputstring, document)