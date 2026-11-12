from fit_edit import fit_candidate
from engine import *
import sys

name, source, random_source, configfile, seed = sys.argv[1:]
config = json.loads(Path(configfile).read_text())
fit_candidate(name, source, config, random_source, int(seed))
