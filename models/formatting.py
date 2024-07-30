
import numpy as np
from mmpose.registry import TRANSFORMS
from mmpose.datasets.transforms import PackPoseInputs

@TRANSFORMS.register_module()
class PackPoseInputsWoImage(PackPoseInputs):
    def transform(self, results: dict) -> dict:
        results = results.copy()
        if 'img' not in results:
            h, w = (256,256)
            results['img'] = np.zeros((h,w,3),dtype = np.uint8)
            results['transformed_keypoints'] = results['keypoints']
        return super().transform(results)