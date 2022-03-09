#!/usr/bin/env python

import unittest
import TestImageLoadingAndDataExtraction
import TestImageConversion

class RegionProposalGeneratorTestCase( unittest.TestCase ):
    def checkVersion(self):
        import RegionProposalGenerator

testSuites = [unittest.makeSuite(RegionProposalGeneratorTestCase, 'test')] 

for test_type in [
            TestImageLoadingAndDataExtraction,
            TestImageConversion,
    ]:
    testSuites.append(test_type.getTestSuites('test'))


def getTestDirectory():
    try:
        return os.path.abspath(os.path.dirname(__file__))
    except:
        return '.'

import os
os.chdir(getTestDirectory())

runner = unittest.TextTestRunner()
runner.run(unittest.TestSuite(testSuites))
