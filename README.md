# MAD Attack Tool

This is a tool for generating examples for MAD subjective testing.  The tool is programmed in a course project.  The library is located at `madatk/`

The module `pytorch_ssim` is from <https://github.com/Po-Hsun-Su/pytorch-ssim>

The module `pytorch_msssim` is from <https://github.com/VainF/pytorch-msssim>

Example (put LIVE in place): 

```sh
./main.py ssim mse ./test/LIVE_779/wn/ ./test/LIVE_779/refimgs/ ./test/LIVE_mad/wn/ -l ./test/LIVE_779/wn/info.txt -i 200 > ./test/LIVE_mad/wn/info.txt 
```

