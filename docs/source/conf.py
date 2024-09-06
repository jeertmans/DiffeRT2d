# type: ignore
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import functools
import inspect
from datetime import date
from textwrap import dedent

import qtgallery
from sphinx.ext.autodoc import between
from sphinx.util.inspect import isclassmethod

from differt2d.__version__ import __version__

project = "DiffeRT2d"
copyright = f"2023-{date.today().year}, Jérome Eertmans"
author = "Jérome Eertmans"
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Built-in
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # Additional
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "qtgallery",
]

add_module_names = False
add_function_parentheses = False

rst_prolog = """
.. role:: python(code)
    :language: python
"""

autodoc_member_order = "bysource"
autodoc_typehints = "description"
typehints_defaults = "comma"

# -- MyST-nb and MyST-parser settings

myst_heading_anchors = 3
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "html_admonition",
]
nb_render_image_options = {"align": "center"}
nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = ["examples_gallery/*.ipynb"]

# Removes the 'package.module' part from package.module.Class
add_module_names = False

show_warning_types = True
suppress_warnings = [
    # WARNING: cannot cache unpickable configuration value: 'sphinx_gallery_conf'
    "config.cache",
]

# generate autosummary even if no references
autosummary_generate = True

sphinx_gallery_conf = {
    "filename_pattern": r"/(plot)|(qt)_",
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "examples_gallery",  # path to where to save gallery generated output
    "image_scrapers": (qtgallery.qtscraper, "matplotlib"),
    "reset_modules": (qtgallery.reset_qapp,),
    "matplotlib_animations": True,
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("differt2d", "equinox", "jax", "jaxtyping", "matplotlib", "optax"),
    "reference_url": {
        "differt2d": None,
    },
    "capture_repr": ("_repr_html_", "__repr__"),
    "compress_images": ("images", "thumbnails"),
    "image_srcset": ["2x"],
    "show_api_usage": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "light_logo": "logo_light_transparent.png",
    "dark_logo": "logo_light_transparent.png",  # We use the same logo
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jeertmans/DiffeRT2d",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/jeertmans/DiffeRT2d/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# -- Intersphinx mapping

intersphinx_mapping = {
    "differt_core": ("https://differt.eertmans.be/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "optax": ("https://optax.readthedocs.io/en/latest", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- OpenGraph settings

ogp_site_url = "https://eertmans.be/DiffeRT2d/"
ogp_use_first_image = True

# -- Bibtex

bibtex_bibfiles = ["references.bib"]

# -- Sphinx App


def get_class_that_defined_method(meth):
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and getattr(meth, "__self__", None) is not None
        and getattr(meth.__self__, "__class__", None)
    ):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, "__func__", meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
            None,
        )
        if isinstance(cls, type):
            return cls
    return getattr(meth, "__objclass__", None)  # handle special descriptor objects


def get_parent_method(obj):
    cls = get_class_that_defined_method(obj)

    if cls is None:
        return None

    for base in cls.__bases__:
        if hasattr(base, obj.__name__):
            return getattr(base, obj.__name__)

    return None


def get_doc(obj):
    doc = obj.__doc__ or ""

    if "differt2d" not in obj.__module__:
        return doc

    if parent := get_parent_method(obj):
        doc = get_doc(parent) + doc

    return doc


def merge_documentation_from_parent(app, what, name, obj, options, lines):
    if what in ["method", "function"]:
        lines[:] = dedent(get_doc(obj)).splitlines()


def unskip_jitted(app, what, name, obj, skip, options):
    """
    Methods that are both jitted and overloading (e.g., from a protocol) are
    skipped, which should not be the case.
    """
    if skip and what == "class" and "Pjit" in repr(obj):
        obj.__doc__ = get_doc(obj._fun)
        return obj.__doc__ == ""


NOTE_ABOUT_ABSTRACT = r"""
.. warning::
    This method is abstract and must be implemented by any of its subsclasses.
""".splitlines()


def add_note_about_abstract(app, what, name, obj, options, lines):
    if getattr(obj, "__isabstractmethod__", False):
        lines.extend(NOTE_ABOUT_ABSTRACT)


def is_singledispatch_classmethod(obj):
    return hasattr(obj, "register") and isclassmethod(obj.__wrapped__, None)


def patch_singledispatch_classmethod_signature(app, obj, bound_method):
    # TODO: fix [source] not appearing
    if bound_method and is_singledispatch_classmethod(obj):
        original = obj.__wrapped__.__func__
        obj.__wrapped__ = original
        for keyword in dir(original):
            if value := getattr(original, keyword, None):
                try:
                    setattr(obj, keyword, value)
                except:  # noqa: E722
                    pass  # Cannot set this


def setup(app):
    app.connect(
        "autodoc-skip-member",
        unskip_jitted,
    )
    app.connect(
        "autodoc-process-docstring",
        merge_documentation_from_parent,
        priority=498,
    )
    app.connect(
        "autodoc-process-docstring",
        between(r".*#.*doc\s*:\s*hide", keepempty=True, exclude=True),
    )
    app.connect(
        "autodoc-process-docstring",
        add_note_about_abstract,
        priority=501,
    )
    app.connect(
        "autodoc-before-process-signature",
        patch_singledispatch_classmethod_signature,
    )
