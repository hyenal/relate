# Video Evaluation with Fréchet Video Distance (FVD)

We provide a copy of the official implementation of [FVD](https://github.com/google-research/google-research/tree/master/frechet_video_distance) to facilitate the reproduction of our results with RELATE's temporal extension.
Please cite the authors of FVD, if you use their code for video evaluation:
```
@article{unterthiner2018towards,
  title={Towards Accurate Generative Models of Video: A New Metric \& Challenges},
  author={Unterthiner, Thomas and van Steenkiste, Sjoerd and Kurach, Karol and Marinier, Raphael and Michalski, Marcin and Gelly, Sylvain},
  journal={arXiv preprint arXiv:1812.01717},
  year={2018}
}
```

To minimize compatibility issues with the remainder of the RELATE codebase, we provide a separate conda environment for the FVD computation [`fvd_environment.yml`](fvd_environment.yml).
Please build it separately using:
```bash
conda env create -f fvd_environment.yml
conda activate fvd
```

After the environment is set up, you can compute the FVD scores for two video directories using:
```bash
(fvd) user@host:~$ python compute_fvd.py path/to/sampled_videos_from_model path/to/test_videos_from_dataset
```
Please note that the videos need to be in GIF format in order to be recognized by the script!
