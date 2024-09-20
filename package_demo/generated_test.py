from codebase import *
from codebase import *
import unittest
from codebase import prime_length

class TestCodebase(unittest.TestCase):

    def test_prime_length_empty_string(self):
        """Test case to check if an empty string returns False"""
        self.assertFalse(prime_length(''))

    def test_prime_length_single_character(self):
        """Test case to check if a single character string returns False"""
        self.assertFalse(prime_length('a'))

    def test_prime_length_prime_length_string(self):
        """Test case to check if a prime length string returns True"""
        self.assertTrue(prime_length('Hello'))

    def test_prime_length_non_prime_length_string(self):
        """Test case to check if a non-prime length string returns False"""
        self.assertFalse(prime_length('orange'))

    def test_prime_length_prime_length_palindrome(self):
        """Test case to check if a prime length palindrome returns True"""
        self.assertTrue(prime_length('abcdcba'))

    def test_prime_length_non_prime_length_palindrome(self):
        """Test case to check if a non-prime length palindrome returns False"""
        self.assertTrue(prime_length('kittens'))

if __name__ == '__main__':
    unittest.main()