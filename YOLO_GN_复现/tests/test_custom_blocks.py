import unittest

import torch

from models.ghost_bifpn import BiFPNP4


class BiFPNP4Test(unittest.TestCase):
    def test_fuses_multiscale_features_at_p4_and_learns_fusion_weights(self):
        module = BiFPNP4([8, 16, 32], 16)
        p3 = torch.randn(2, 8, 80, 80, requires_grad=True)
        p4 = torch.randn(2, 16, 40, 40, requires_grad=True)
        p5 = torch.randn(2, 32, 20, 20, requires_grad=True)

        output = module([p3, p4, p5])
        self.assertEqual(tuple(output.shape), (2, 16, 40, 40))
        output.square().mean().backward()

        self.assertIsNotNone(module.fusion_weights.grad)
        self.assertTrue(torch.isfinite(module.fusion_weights.grad).all())
        self.assertTrue(torch.isfinite(p3.grad).all())
        self.assertTrue(torch.isfinite(p4.grad).all())
        self.assertTrue(torch.isfinite(p5.grad).all())


if __name__ == '__main__':
    unittest.main()
import unittest

import torch
from ultralytics import YOLO

from models.ghost_bifpn import BiFPNP4


class YOLOGNBiFPNBuildTest(unittest.TestCase):
    def test_yolo11s_custom_yaml_builds_and_returns_three_detection_scales(self):
        yolo = YOLO('models/yolo11s_gn_bifpn.yaml', task='detect')
        detector = yolo.model.eval()
        self.assertIsInstance(detector.model[-2], BiFPNP4)

        with torch.no_grad():
            output = detector(torch.zeros(1, 3, 640, 640))
        predictions = output[0] if isinstance(output, tuple) else output
        self.assertEqual(tuple(predictions.shape), (1, 6, 8400))


if __name__ == '__main__':
    unittest.main()

class EnhancementTest(unittest.TestCase):
    def test_clahe_contrast_stretch_preserves_rgb_image_shape_and_range(self):
        import numpy as np
        from scripts.enhance_dataset import enhance_image

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[12:52, 14:50] = (42, 74, 96)
        output = enhance_image(image)

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertGreater(int(output.max()), int(image.max()))
        self.assertLessEqual(int(output.min()), 255)
