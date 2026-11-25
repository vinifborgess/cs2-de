"""Test package.

The data lake location is read when ``cs2_pipeline.config`` is first imported, so it is
pointed at a scratch directory HERE, before any test module (and through it any pipeline
module) is imported. Doing it inside a single test module is not enough: whichever module
the runner imports first wins, and a module that imports the pipeline earlier would make
the tests write into the real ``data/`` lake.
"""
import os
import tempfile

os.environ.setdefault("CS2_TEST_SCRATCH", tempfile.mkdtemp(prefix="cs2-test-"))
os.environ["CS2_DATA_DIR"] = os.environ["CS2_TEST_SCRATCH"]
