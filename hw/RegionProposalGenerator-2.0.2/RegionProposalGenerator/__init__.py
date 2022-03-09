#!/usr/bin/env python

import sys

if sys.version_info[0] == 3:
    from RegionProposalGenerator.RegionProposalGenerator import __version__
    from RegionProposalGenerator.RegionProposalGenerator import __author__
    from RegionProposalGenerator.RegionProposalGenerator import __date__
    from RegionProposalGenerator.RegionProposalGenerator import __url__
    from RegionProposalGenerator.RegionProposalGenerator import __copyright__
    from RegionProposalGenerator.RegionProposalGenerator import RegionProposalGenerator
else:
    from RegionProposalGenerator import __version__
    from RegionProposalGenerator import __author__
    from RegionProposalGenerator import __date__
    from RegionProposalGenerator import __url__
    from RegionProposalGenerator import __copyright__
    from RegionProposalGenerator import RegionProposalGenerator





