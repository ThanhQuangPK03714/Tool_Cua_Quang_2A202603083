"""Unit test: mã hóa mask (RLE, polygon) và kích thước mask."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mask_processor import mask_to_polygons, mask_to_rle, resize_mask


class TestMaskEncoding(unittest.TestCase):
    def setUp(self):
        # Tạo mask nhị phân đơn giản: hình chữ nhật 10x10 ở góc trên trái
        self.mask = np.zeros((100, 100), dtype=np.uint8)
        self.mask[10:20, 10:20] = 1

    def test_mask_to_rle(self):
        rle = mask_to_rle(self.mask)
        self.assertIn("size", rle)
        self.assertIn("counts", rle)
        self.assertEqual(rle["size"], [100, 100])

    def test_mask_to_polygons(self):
        polys = mask_to_polygons(self.mask)
        self.assertGreaterEqual(len(polys), 1)
        # Mỗi polygon phải là list các số float (x1, y1, x2, y2, ...)
        for poly in polys:
            self.assertEqual(len(poly) % 2, 0)
            self.assertGreaterEqual(len(poly), 6)  # ít nhất 3 điểm

    def test_resize_mask(self):
        resized = resize_mask(self.mask, (50, 50))
        self.assertEqual(resized.shape, (50, 50))
        # Sau khi resize, vẫn có vùng trắng (mask được giữ)
        self.assertGreater(resized.sum(), 0)

    def test_mask_binary(self):
        self.assertTrue(np.all((self.mask == 0) | (self.mask == 1)))


if __name__ == "__main__":
    unittest.main()
