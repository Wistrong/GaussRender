# GaussRender
This is the official implementation of: "GaussRender: Learning 3D Occupancy with Gaussian Rendering."

## Abstract
Understanding the 3D geometry and semantics of driving scenes is critical for developing of safe autonomous vehicles. While 3D occupancy models are typically trained using voxel-based supervision with standard losses (e.g., cross-entropy, Lovasz, dice), these approaches treat voxel predictions independently, neglecting their spatial relationships. In this paper, we propose GaussRender, a plug-and-play 3D-to-2D reprojection loss that enhances voxel-based supervision. Our method projects 3D voxel representations into arbitrary 2D perspectives and leverages Gaussian splatting as an efficient, differentiable rendering proxy of voxels, introducing spatial dependencies across projected elements. This approach improves semantic and geometric consistency, handles occlusions more efficiently, and requires no architectural modifications. Extensive experiments on multiple benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) demonstrate consistent performance gains across various 3D occupancy models (TPVFormer, SurroundOcc, Symphonies), highlighting the robustness and versatility of our framework. The code is available at this https URL.

## News
- Training and evaluation code with checkpoints will be uploaded soon ⚡
- Paper uploaded on arxiv 🚀

## Cite

```bash
@misc{chambon2025gaussrenderlearning3doccupancy,
      title={GaussRender: Learning 3D Occupancy with Gaussian Rendering}, 
      author={Loick Chambon and Eloi Zablocki and Alexandre Boulch and Mickael Chen and Matthieu Cord},
      year={2025},
      eprint={2502.05040},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.05040}, 
}
```
