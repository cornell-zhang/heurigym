from unittest import TestCase
import pytoulbar2

class TestExtension(TestCase):
   def test_1(self):
      myCFN = pytoulbar2.CFN(2)
      res = myCFN.Solve()
      self.assertEqual(res[0],[])
      self.assertEqual(res[1],0.0)
      self.assertEqual(res[2],1)
 
