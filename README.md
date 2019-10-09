# Python Fingerprint Recognition

Fingerprint recognition with SKimage and OpenCV

Requirements:
- NumPy
- SKimage
- OpenCV2


Works by extracting minutiae points using harris corner detection.

Uses SIFT (ORB) go get formal descriptors around the keypoints with brute-force hamming distance and then analyzes the returned matches using thresholds.

Usage:

1. Place 2 fingerprint images that you want to compare inside the database folder
2. Pass the names of the images as arguments in the console

## Dockerfile

If you don't want to install the libraries, or want a easier way to test the application you can follow the commands:

```shell
docker build -t <name_of_your_choice> .

docker run -it <name_of_your_choice> <fingerprint_1> <fingerprint_2>
```

NOTE: the fingerprints must be in the `/database` folder
