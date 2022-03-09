import RegionProposalGenerator
import unittest

class TestImageLoadingAndDataExtraction(unittest.TestCase):

    def setUp(self):
        self.rpg = RegionProposalGenerator.RegionProposalGenerator(data_image="halfsun.jpg")
        self.rpg.extract_data_pixels_in_bb("halfsun.jpg", [5,5,6,6])

    def test_loaded_image(self):
        print("testing image loading and pixel data extraction")
        self.rpg.displayImage(self.rpg.data_im)
        width,height = self.rpg.data_im.size
        val = self.rpg.extract_data_pixels_in_bb("halfsun.jpg", [5,5,6,6])
        self.assertEqual(width, 150)
        self.assertEqual(height, 49)
        self.assertIn( val[0][0][0], range(45,50) )
        self.assertIn( val[0][0][1], range(4,9) )
        self.assertIn( val[0][0][2], range(6,9) )

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestImageLoadingAndDataExtraction, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

