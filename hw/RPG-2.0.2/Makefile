# GNU -*- makefile -*-

#VERSION := ${shell python -c "import RegionProposalGenerator;print RegionProposalGenerator.__version__"}
VERSION := ${shell python -c "import os; os.chdir('RegionProposalGenerator'); import RegionProposalGenerator; print(RegionProposalGenerator.__version__)"}

default:
	@echo
	@echo "  *** Welcome to RegionProposalGenerator ${VERSION} ***"
	@echo
	@echo "  docs   -  Build documentation (html)"
	@echo "  help   -  Open the documentation"
	@echo
	@echo "  clean  -  Remove temporary files"
	@echo "  test   -  Run the unittests"
	@echo "  check  -  Look for rough spots"
	@echo "  sdist  -  Build a source distribution tar ball"

docs:
	pydoc -w RegionProposalGenerator/RegionProposalGenerator.py

help:
	open RegionProposalGenerator-${VERSION}.html

clean:
	rm -f *.pyc *~

real-clean: clean
	rm -f MANIFEST  *.html RegionProposalGenerator-py.info
	rm -rf build dist

# Run the unittest
test:
	@echo
	@echo Testing...
	@echo
	./TestRegionProposalGenerator/Test.py 

sdist: test
	@echo
	@echo Building a source distribution...
	@echo
	./setup.py sdist --formats=gztar,bztar

# Look for rough spots
check:
	@grep -n FIX *.py *.in PKG-INFO Makefile | grep -v grep
	@echo
	pychecker RegionProposalGenerator


