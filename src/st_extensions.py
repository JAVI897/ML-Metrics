"""Extensions of the streamlit api
For now these are hacks and hopefully a lot of them will be removed again as the streamlit api is
extended"""
import logging
import sys
import importlib
import streamlit as st

import config

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def write_page(page):  # pylint: disable=redefined-outer-name
    """Writes the specified page/module
    Our multipage app is structured into sub-files with a `def write()` function
    Arguments:
        page {module} -- A module with a 'def write():' function
    """
    if config.DEBUG:
        logging.info("1. Writing: %s", page)
        logging.info("2. In sys.modules: %s", page in sys.modules)
        try:
            importlib.import_module(page.__name__)
            importlib.reload(page)
        except ImportError as identifier:
            logging.info("3. Writing: %s", page)
            logging.info("4. In sys.modules: %s", page in sys.modules)
    page.write()