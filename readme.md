
# Official implementation of: *GaussRender: Learning 3D Occupancy with Gaussian Rendering.*

> [**GaussRender: Learning 3D Occupancy with Gaussian Rendering.**](https://arxiv.org/abs/2502.05040)<br>
> [Loick Chambon](https://loickch.github.io/), [Eloi Zablocki](https://scholar.google.fr/citations?user=dOkbUmEAAAAJ&hl=fr), [Alexandre Boulch](https://boulch.eu/), [Mickael Chen](https://sites.google.com/view/mickaelchen/), [Matthieu Cord](https://cord.isir.upmc.fr/).<br> Valeo AI, Sorbonne University

<table>
  <tr>
    <td align="center" width="50%">
      <img src="asset/demo_scene_0003.gif" width="100%">
    </td>
    <td align="center" width="50%">
      <img src="asset/demo_scene_0013.gif" width="100%">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <em>GaussRender is a 3D Occupancy module that can be plugged into any 3D Occupancy model to enhance its predictions and ensure 2D-3D consistency while improving mIoU, IoU, and RayIoU.</em>
    </td>
  </tr>
</table>


# Abstract

*Understanding the 3D geometry and semantics of driving scenes is critical for developing of safe autonomous vehicles. While 3D occupancy models are typically trained using voxel-based supervision with standard losses (e.g., cross-entropy, Lovasz, dice), these approaches treat voxel predictions independently, neglecting their spatial relationships. In this paper, we propose GaussRender, a plug-and-play 3D-to-2D reprojection loss that enhances voxel-based supervision. Our method projects 3D voxel representations into arbitrary 2D perspectives and leverages Gaussian splatting as an efficient, differentiable rendering proxy of voxels, introducing spatial dependencies across projected elements. This approach improves semantic and geometric consistency, handles occlusions more efficiently, and requires no architectural modifications. Extensive experiments on multiple benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) demonstrate consistent performance gains across various 3D occupancy models (TPVFormer, SurroundOcc, Symphonies), highlighting the robustness and versatility of our framework.*


<table>
  <tr>
    <td align="center">
      <img src="asset/pipeline.jpg" width="800">
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>GaussRender can be plugged to any model. The core idea is to transform voxels into gaussians before performing a depth and a semantic rendering.</em>
    </td>
  </tr>
</table>

## Updates:
* 【07/02/2025】 [GaussRender](https://arxiv.org/abs/2502.05040) is on arxiv.

# 🚀 Main results

## 🔥 3D Occupancy
GaussRender can be plugged into any 3D model. We have dedicated experiments on multiple 3D benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) and on multiple models (TPVFormer, SurroundOcc, Symphonies) to evaluate its performance.

### Occ3d-nuScenes

<div align="center">
<table border="1">
  <caption><i>3D mIoU and IoU of several models on the Occ3D-nuScenes dataset. Best result in <span style="color:blue;">blue</span>, second best in <span style="color:green;">green</span>.</i></caption>
    <tr>
        <th>Models</th>
        <th><a href="https://arxiv.org/abs/2502.05040">TPVFormer (ours) </a></th>
        <th><a href="https://arxiv.org/abs/2302.07817">TPVFormer </a></th>
        <th><a href="https://arxiv.org/abs/2502.05040">SurroundOcc (ours) </a></th>
        <th><a href="https://arxiv.org/abs/2303.09551">SurroundOcc </a></th>
        <th><a href="https://arxiv.org/abs/2304.05316">OccFormer</a></th>
        <th><a href="https://arxiv.org/abs/2309.09502">RenderOcc</a></th>
    </tr>
    <tr>
      <th> Type </th>
      <th> w/ GaussRender </th>
      <th> base </th>
      <th> w/ GaussRender </th>
      <th> base </th>
      <th> base </th>
      <th> base </th>
    </tr>
    <tr class="highlight-column">
        <td>mIoU</td>
        <td style="color:blue; font-weight:bold;">30.48</td>
        <td>27.83</td>
        <td style="color:green; font-weight:bold;">30.38</td>
        <td>29.21</td>
        <td>21.93</td>
        <td>26.11</td>
    </tr>
    <tr class="highlight-column">
        <td>RayIoU</td>
        <td style="color:blue; font-weight:bold;">38.3</td>
        <td>37.2</td>
        <td style="color:green; font-weight:bold;">37.5</td>
        <td>35.5</td>
        <td>-</td>
        <td>19.5</td>
    </tr>
</table>
</div>

### SurroundOcc-nuScenes

<div align="center">
<table border="1">
  <caption><i>3D mIoU and IoU of several models on the SurroundOcc-nuScenes dataset. Best result in <span style="color:blue;">blue</span>, second best in <span style="color:green;">green</span>.</i></caption>
    <tr>
        <th>Models</th>
        <th><a href="https://arxiv.org/abs/2502.05040">TPVFormer (ours) </a></th>
        <th><a href="https://arxiv.org/abs/2302.07817">TPVFormer </a></th>
        <th><a href="https://arxiv.org/abs/2502.05040">SurroundOcc (ours) </a></th>
        <th><a href="https://arxiv.org/abs/2303.09551">SurroundOcc </a></th>
        <th><a href="https://arxiv.org/abs/2304.05316">OccFormer</a></th>
        <th><a href="https://arxiv.org/abs/2412.04384">GaussianFormerv2</a></th>
    </tr>
    <tr>
      <th> Type </th>
      <th> w/ GaussRender </th>
      <th> base </th>
      <th> w/ GaussRender </th>
      <th> base </th>
      <th> base </th>
      <th> base </th>
    </tr>
    <tr class="highlight-column">
        <td>IoU</td>
        <td style="color:green; font-weight:bold;">32.05</td>
        <td>30.86</td>
        <td style="color:blue; font-weight:bold;">32.61</td>
        <td>31.49</td>
        <td>31.39</td>
        <td>30.56</td>
    </tr>
    <tr class="highlight-column">
        <td>mIoU</td>
        <td style="color:green; font-weight:bold;">20.58</td>
        <td>17.10</td>
        <td style="color:blue; font-weight:bold;">20.82</td>
        <td>20.30</td>
        <td>19.03</td>
        <td>20.02</td>
    </tr>
</table>
</div>

### SSCBench-KITTI360

<div align="center">
<table border="1">
  <caption><i>3D mIoU and IoU of several models on the SSCBench-KITTI360 dataset. Best result in <span style="color:blue;">blue</span>, second best in <span style="color:green;">green</span>.</i></caption>
    <tr>
        <th>Models</th>
        <th><a href="https://arxiv.org/abs/2502.05040">SurroundOcc (ours) </a></th>
        <th><a href="https://arxiv.org/abs/2303.09551">SurroundOcc </a></th>
        <th><a href="https://arxiv.org/abs/2502.05040">Symphonies (ours) </a></th>
        <th><a href="https://arxiv.org/abs/2306.15670">Symphonies </a></th>
        <th><a href="https://arxiv.org/abs/2304.05316">OccFormer</a></th>
        <th><a href="https://arxiv.org/abs/2112.00726">MonoScene</a></th>
    </tr>
    <tr>
      <th> Type </th>
      <th> w/ GaussRender </th>
      <th> base </th>
      <th> w/ GaussRender </th>
      <th> base </th>
      <th> base </th>
      <th> base </th>
    </tr>
    <tr class="highlight-column">
        <td>IoU</td>
        <td>38.62</td>
        <td>38.51</td>
        <td style="color:blue; font-weight:bold;">44.08</td>
        <td style="color:green; font-weight:bold;">43.40</td>
        <td>40.27</td>
        <td>37.87</td>
    </tr>
    <tr class="highlight-column">
        <td>mIoU</td>
        <td>13.34</td>
        <td>13.08</td>
        <td style="color:blue; font-weight:bold;">18.11</td>
        <td style="color:green; font-weight:bold;">17.82</td>
        <td>13.81</td>
        <td>12.31</td>
    </tr>
</table>
</div>

# 🔨 Setup <a name="setup"></a>
Soon

## 👍 Acknowledgements

Many thanks to these excellent open source projects:
* [GaussianFormer](https://github.com/huang-yh/GaussianFormer)
* [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
* [TPVFormer](https://github.com/wzzheng/TPVFormer)

## ❤️  Other repository
If you liked our work, do not hesitate to also see:
* [PointBeV](https://github.com/valeoai/PointBeV): sparse BeV 2D segmentation.

## ✏️ Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entry and putting a star on this repository.

```
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
