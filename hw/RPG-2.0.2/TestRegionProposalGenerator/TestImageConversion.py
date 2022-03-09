import RegionProposalGenerator
import os
import unittest

class TestImageConversion(unittest.TestCase):

    def setUp(self):
        self.rpg = RegionProposalGenerator.RegionProposalGenerator()
        self.rpg.working_with_hsv_color_space("halfsun.jpg", test=True)

    def test_image_conversion(self):
        available = False
        print("testing image conversion")
        hsv_image = self.rpg.working_with_hsv_color_space("halfsun.jpg", test=True)
        if os.path.exists("hsv_arr.npy"):
            available = True
        self.assertEqual(available, True)
        os.remove("hsv_arr.npy")

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestImageConversion, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

