import traceback
from pathlib import Path

here = Path(__file__).parent


def refresh_entry_points():
    """\
    Under some circumstances, (e.g. when installing a PEP 517 package via pip),
    pkg_resources.working_set.entries is stale. This tries to fix that.
    """
    try:
        import sys
        import pkg_resources

        ws: pkg_resources.WorkingSet = pkg_resources.working_set
        for entry in sys.path:
            ws.add_entry(entry)
    except Exception:
        pass


try:
    from setuptools_scm import get_version
    import pytoml

    proj = pytoml.loads((here.parent / 'pyproject.toml').read_text())
    metadata = proj['tool']['flit']['metadata']

    refresh_entry_points()
    __version__ = get_version(root='..', relative_to=__file__)
    __author__ = metadata['author']
    __email__ = metadata['author-email']
except (ImportError, LookupError, FileNotFoundError):
    from ._compat import pkg_metadata

    metadata = pkg_metadata(here.name)
    __version__ = metadata['Version']
    __author__ = metadata['Author']
    __email__ = metadata['Author-email']


def within_flit():
    for frame in traceback.extract_stack():
        if frame.name == 'get_docstring_and_version_via_import':
            return True
    return False
