import os
import sys

# Make "import options_engine" work from the tests without installing the
# package, by putting src/ on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
