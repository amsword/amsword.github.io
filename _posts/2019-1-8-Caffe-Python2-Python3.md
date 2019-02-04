---
layout: post
title: Caffe Performance
---

# Single GPU
## Caffe exe
```bash
./src/CCSCaffe/build/tools/caffe.bin train \
        --solver output/TaxInsideV3_1_darknet19_448_C_Init.best_model7428_maxIter.10eEffectBatchSize128_FixParam.dark6a.leaky_bb_only/solver.prototxt \
        --gpu 4
```

python 2: 24.1
python 3: 16.5
System python 3.5: 16.7

## Caffe python
```bash
python scripts/qd_common.py -p "{'type': 'caffe_train', \
               'gpu': 4, \
               'solver_prototxt': 'output/TaxInsideV3_1_darknet19_448_C_Init.best_model7428_maxIter.10eEffectBatchSize128_FixParam.dark6a.leaky_bb_only/solver.prototxt'}"
```
python 2: 24.4
Anaconda python 3.6: 31.3
system python 3.5: 23.8

# Single GPU, but no data loading
We remove data loading by directly seconding the current buffer to the output
of the datalayer without loading the data from the pre-fetched buffer.
```bash
python scripts/qd_common.py -p "{'type': 'caffe_train', \
               'gpu': 4, \
               'solver_prototxt': 'output/TaxInsideV3_1_darknet19_448_C_Init.best_model7428_maxIter.10eEffectBatchSize128_FixParam.dark6a.leaky_bb_only/solver.prototxt'}"
```
python 2: 14.1
Anaconda python 3.6: 14.2


# 4 GPU
## exe
``` bash
./src/CCSCaffe/build/tools/caffe.bin train \
        --solver output/TaxInsideV3_1_darknet19_448_C_Init.best_model7428_maxIter.10eEffectBatchSize128_FixParam.dark6a.leaky_bb_only/solver.prototxt \
        --gpu 4,5,6,7
```
python 2: 26.0
anaconda 3.6: 25.9
system python 3.5: 25.0 s

## pycaffe
```bash
python ./scripts/qd_common.py -p "{'type': 'parallel_train',\
    'solver': './output/voc20_darknet19_448_debug_noreorg_extraconv2/solver.prototxt', \
    'snapshot': null, \
    'weights': null, \
    'gpus': [4,5,6,7]}"
```
python2: 25.3
anaconda 3.6: 41.3
python 3.5: 25.0

# Single GPU but with omp thread change, python interface
python 2: 23.8
Anaconda python 3.6: 22.5

# Four GPU, with omp thread change, python interface
python 2: 23.5
python 3: 23.3
Anaconda python 3.6: 26.2


## Shared Library
python2
```bash
linux-vdso.so.1 =>  (0x00007fff4ad0b000)
libcaffe.so.1.0.0-rc5 => /tmp/code/quickdetection.2.7/src/CCSCaffe/python/caffe/../../build/lib/libcaffe.so.1.0.0-rc5 (0x00007fb536691000)
libglog.so.0 => /usr/lib/x86_64-linux-gnu/libglog.so.0 (0x00007fb536462000)
libprotobuf.so.9 => /usr/lib/x86_64-linux-gnu/libprotobuf.so.9 (0x00007fb536144000)
libboost_system.so.1.58.0 => /usr/lib/x86_64-linux-gnu/libboost_system.so.1.58.0 (0x00007fb535f40000)
libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fb535bb4000)
libboost_python-py27.so.1.58.0 => /usr/lib/x86_64-linux-gnu/libboost_python-py27.so.1.58.0 (0x00007fb535968000)
libpython2.7.so.1.0 => /usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0 (0x00007fb5353da000)
libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fb5351c3000)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fb534df9000)
libcudart.so.9.0 => /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so.9.0 (0x00007fb534b8c000)
libcublas.so.9.0 => /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcublas.so.9.0 (0x00007fb531756000)
libcurand.so.9.0 => /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcurand.so.9.0 (0x00007fb52d7f2000)
libgflags.so.2 => /usr/lib/x86_64-linux-gnu/libgflags.so.2 (0x00007fb52d5d1000)
libboost_filesystem.so.1.58.0 => /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.58.0 (0x00007fb52d3b9000)
libhdf5_serial_hl.so.10 => /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10 (0x00007fb52d199000)
libhdf5_serial.so.10 => /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10 (0x00007fb52ccfb000)
libleveldb.so.1 => /usr/lib/x86_64-linux-gnu/libleveldb.so.1 (0x00007fb52caa1000)
liblmdb.so.0 => /usr/lib/x86_64-linux-gnu/liblmdb.so.0 (0x00007fb52c88c000)
libopencv_core.so.2.4 => /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4 (0x00007fb52c462000)
libopencv_highgui.so.2.4 => /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4 (0x00007fb52c211000)
libopencv_imgproc.so.2.4 => /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4 (0x00007fb52bd86000)
libboost_thread.so.1.58.0 => /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.58.0 (0x00007fb52bb60000)
libcudnn.so.7 => /usr/lib/x86_64-linux-gnu/libcudnn.so.7 (0x00007fb51944e000)
libnccl.so.2 => /usr/lib/x86_64-linux-gnu/libnccl.so.2 (0x00007fb514749000)
libcblas.so.3 => /usr/lib/libcblas.so.3 (0x00007fb514527000)
libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fb51421e000)
libgomp.so.1 => /usr/lib/x86_64-linux-gnu/libgomp.so.1 (0x00007fb513fef000)
libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fb513dd2000)
libunwind.so.8 => /usr/local/lib/libunwind.so.8 (0x00007fb513bba000)
libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fb5139a0000)
/lib64/ld-linux-x86-64.so.2 (0x00007fb537f06000)
libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fb51379c000)
libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007fb513599000)
librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007fb513391000)
libsz.so.2 => /usr/lib/x86_64-linux-gnu/libsz.so.2 (0x00007fb51318e000)
libsnappy.so.1 => /usr/lib/x86_64-linux-gnu/libsnappy.so.1 (0x00007fb512f86000)
libGL.so.1 => /usr/lib/nvidia-384/libGL.so.1 (0x00007fb512c44000)
libtbb.so.2 => /usr/lib/x86_64-linux-gnu/libtbb.so.2 (0x00007fb512a07000)
libjpeg.so.8 => /usr/lib/x86_64-linux-gnu/libjpeg.so.8 (0x00007fb5127ae000)
libpng12.so.0 => /lib/x86_64-linux-gnu/libpng12.so.0 (0x00007fb512589000)
libtiff.so.5 => /usr/lib/x86_64-linux-gnu/libtiff.so.5 (0x00007fb512315000)
libjasper.so.1 => /usr/lib/x86_64-linux-gnu/libjasper.so.1 (0x00007fb5120c0000)
libIlmImf-2_2.so.22 => /usr/lib/x86_64-linux-gnu/libIlmImf-2_2.so.22 (0x00007fb511bf2000)
libHalf.so.12 => /usr/lib/x86_64-linux-gnu/libHalf.so.12 (0x00007fb5119af000)
libgtk-x11-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgtk-x11-2.0.so.0 (0x00007fb511364000)
libgdk-x11-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgdk-x11-2.0.so.0 (0x00007fb5110af000)
libgobject-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0 (0x00007fb510e5c000)
libglib-2.0.so.0 => /lib/x86_64-linux-gnu/libglib-2.0.so.0 (0x00007fb510b4b000)
libgtkglext-x11-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libgtkglext-x11-1.0.so.0 (0x00007fb510947000)
libgdkglext-x11-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libgdkglext-x11-1.0.so.0 (0x00007fb5106e3000)
libdc1394.so.22 => /usr/lib/x86_64-linux-gnu/libdc1394.so.22 (0x00007fb51046d000)
libv4l1.so.0 => /usr/lib/x86_64-linux-gnu/libv4l1.so.0 (0x00007fb510267000)
libavcodec-ffmpeg.so.56 => /usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56 (0x00007fb50ee36000)
libavformat-ffmpeg.so.56 => /usr/lib/x86_64-linux-gnu/libavformat-ffmpeg.so.56 (0x00007fb50ea37000)
libavutil-ffmpeg.so.54 => /usr/lib/x86_64-linux-gnu/libavutil-ffmpeg.so.54 (0x00007fb50e7c8000)
libswscale-ffmpeg.so.3 => /usr/lib/x86_64-linux-gnu/libswscale-ffmpeg.so.3 (0x00007fb50e539000)
libatlas.so.3 => /usr/lib/libatlas.so.3 (0x00007fb50df9b000)
libgfortran.so.3 => /usr/lib/x86_64-linux-gnu/libgfortran.so.3 (0x00007fb50dc6a000)
liblzma.so.5 => /lib/x86_64-linux-gnu/liblzma.so.5 (0x00007fb50da48000)
libaec.so.0 => /usr/lib/x86_64-linux-gnu/libaec.so.0 (0x00007fb50d840000)
libnvidia-tls.so.384.130 => /usr/lib/nvidia-384/tls/libnvidia-tls.so.384.130 (0x00007fb50d63c000)
libnvidia-glcore.so.384.130 => /usr/lib/nvidia-384/libnvidia-glcore.so.384.130 (0x00007fb50b780000)
libX11.so.6 => /usr/lib/x86_64-linux-gnu/libX11.so.6 (0x00007fb50b446000)
libXext.so.6 => /usr/lib/x86_64-linux-gnu/libXext.so.6 (0x00007fb50b234000)
libjbig.so.0 => /usr/lib/x86_64-linux-gnu/libjbig.so.0 (0x00007fb50b026000)
libIex-2_2.so.12 => /usr/lib/x86_64-linux-gnu/libIex-2_2.so.12 (0x00007fb50ae08000)
libIlmThread-2_2.so.12 => /usr/lib/x86_64-linux-gnu/libIlmThread-2_2.so.12 (0x00007fb50ac01000)
libgmodule-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgmodule-2.0.so.0 (0x00007fb50a9fd000)
libpangocairo-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpangocairo-1.0.so.0 (0x00007fb50a7f0000)
libXfixes.so.3 => /usr/lib/x86_64-linux-gnu/libXfixes.so.3 (0x00007fb50a5ea000)
libatk-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libatk-1.0.so.0 (0x00007fb50a3c5000)
libcairo.so.2 => /usr/lib/x86_64-linux-gnu/libcairo.so.2 (0x00007fb50a0b1000)
libgdk_pixbuf-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgdk_pixbuf-2.0.so.0 (0x00007fb509e8f000)
libgio-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 (0x00007fb509b07000)
libpangoft2-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpangoft2-1.0.so.0 (0x00007fb5098f1000)
libpango-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpango-1.0.so.0 (0x00007fb5096a5000)
libfontconfig.so.1 => /usr/lib/x86_64-linux-gnu/libfontconfig.so.1 (0x00007fb509462000)
libXrender.so.1 => /usr/lib/x86_64-linux-gnu/libXrender.so.1 (0x00007fb509258000)
libXinerama.so.1 => /usr/lib/x86_64-linux-gnu/libXinerama.so.1 (0x00007fb509055000)
libXi.so.6 => /usr/lib/x86_64-linux-gnu/libXi.so.6 (0x00007fb508e45000)
libXrandr.so.2 => /usr/lib/x86_64-linux-gnu/libXrandr.so.2 (0x00007fb508c3a000)
libXcursor.so.1 => /usr/lib/x86_64-linux-gnu/libXcursor.so.1 (0x00007fb508a30000)
libXcomposite.so.1 => /usr/lib/x86_64-linux-gnu/libXcomposite.so.1 (0x00007fb50882d000)
libXdamage.so.1 => /usr/lib/x86_64-linux-gnu/libXdamage.so.1 (0x00007fb50862a000)
libffi.so.6 => /usr/lib/x86_64-linux-gnu/libffi.so.6 (0x00007fb508422000)
libpcre.so.3 => /lib/x86_64-linux-gnu/libpcre.so.3 (0x00007fb5081b2000)
libGLU.so.1 => /usr/lib/x86_64-linux-gnu/libGLU.so.1 (0x00007fb507f43000)
libXmu.so.6 => /usr/lib/x86_64-linux-gnu/libXmu.so.6 (0x00007fb507d2a000)
libpangox-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpangox-1.0.so.0 (0x00007fb507b0a000)
libraw1394.so.11 => /usr/lib/x86_64-linux-gnu/libraw1394.so.11 (0x00007fb5078fb000)
libusb-1.0.so.0 => /lib/x86_64-linux-gnu/libusb-1.0.so.0 (0x00007fb5076e3000)
libv4l2.so.0 => /usr/lib/x86_64-linux-gnu/libv4l2.so.0 (0x00007fb5074d5000)
libswresample-ffmpeg.so.1 => /usr/lib/x86_64-linux-gnu/libswresample-ffmpeg.so.1 (0x00007fb5072b8000)
libva.so.1 => /usr/lib/x86_64-linux-gnu/libva.so.1 (0x00007fb50709c000)
libzvbi.so.0 => /usr/lib/x86_64-linux-gnu/libzvbi.so.0 (0x00007fb506e11000)
libxvidcore.so.4 => /usr/lib/x86_64-linux-gnu/libxvidcore.so.4 (0x00007fb506afd000)
libx265.so.79 => /usr/lib/x86_64-linux-gnu/libx265.so.79 (0x00007fb505ede000)
libx264.so.148 => /usr/lib/x86_64-linux-gnu/libx264.so.148 (0x00007fb505b3a000)
libwebp.so.5 => /usr/lib/x86_64-linux-gnu/libwebp.so.5 (0x00007fb5058de000)
libwavpack.so.1 => /usr/lib/x86_64-linux-gnu/libwavpack.so.1 (0x00007fb5056b5000)
libvpx.so.3 => /usr/lib/x86_64-linux-gnu/libvpx.so.3 (0x00007fb505291000)
libvorbisenc.so.2 => /usr/lib/x86_64-linux-gnu/libvorbisenc.so.2 (0x00007fb504fe8000)
libvorbis.so.0 => /usr/lib/x86_64-linux-gnu/libvorbis.so.0 (0x00007fb504dbd000)
libtwolame.so.0 => /usr/lib/x86_64-linux-gnu/libtwolame.so.0 (0x00007fb504b9a000)
libtheoraenc.so.1 => /usr/lib/x86_64-linux-gnu/libtheoraenc.so.1 (0x00007fb50495b000)
libtheoradec.so.1 => /usr/lib/x86_64-linux-gnu/libtheoradec.so.1 (0x00007fb504741000)
libspeex.so.1 => /usr/lib/x86_64-linux-gnu/libspeex.so.1 (0x00007fb504528000)
libshine.so.3 => /usr/lib/x86_64-linux-gnu/libshine.so.3 (0x00007fb50431b000)
libschroedinger-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libschroedinger-1.0.so.0 (0x00007fb504046000)
libopus.so.0 => /usr/lib/x86_64-linux-gnu/libopus.so.0 (0x00007fb503dfc000)
libopenjpeg.so.5 => /usr/lib/x86_64-linux-gnu/libopenjpeg.so.5 (0x00007fb503bd9000)
libmp3lame.so.0 => /usr/lib/x86_64-linux-gnu/libmp3lame.so.0 (0x00007fb503964000)
libgsm.so.1 => /usr/lib/x86_64-linux-gnu/libgsm.so.1 (0x00007fb503756000)
libcrystalhd.so.3 => /usr/lib/x86_64-linux-gnu/libcrystalhd.so.3 (0x00007fb50353b000)
libssh-gcrypt.so.4 => /usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4 (0x00007fb5032f2000)
librtmp.so.1 => /usr/lib/x86_64-linux-gnu/librtmp.so.1 (0x00007fb5030d6000)
libmodplug.so.1 => /usr/lib/x86_64-linux-gnu/libmodplug.so.1 (0x00007fb502d4b000)
libgme.so.0 => /usr/lib/x86_64-linux-gnu/libgme.so.0 (0x00007fb502afd000)
libbluray.so.1 => /usr/lib/x86_64-linux-gnu/libbluray.so.1 (0x00007fb5028b4000)
libgnutls.so.30 => /usr/lib/x86_64-linux-gnu/libgnutls.so.30 (0x00007fb502584000)
libbz2.so.1.0 => /lib/x86_64-linux-gnu/libbz2.so.1.0 (0x00007fb502374000)
libquadmath.so.0 => /usr/lib/x86_64-linux-gnu/libquadmath.so.0 (0x00007fb502135000)
libxcb.so.1 => /usr/lib/x86_64-linux-gnu/libxcb.so.1 (0x00007fb501f13000)
libfreetype.so.6 => /usr/lib/x86_64-linux-gnu/libfreetype.so.6 (0x00007fb501c69000)
libpixman-1.so.0 => /usr/lib/x86_64-linux-gnu/libpixman-1.so.0 (0x00007fb5019c1000)
libxcb-shm.so.0 => /usr/lib/x86_64-linux-gnu/libxcb-shm.so.0 (0x00007fb5017bd000)
libxcb-render.so.0 => /usr/lib/x86_64-linux-gnu/libxcb-render.so.0 (0x00007fb5015b3000)
libselinux.so.1 => /lib/x86_64-linux-gnu/libselinux.so.1 (0x00007fb501391000)
libresolv.so.2 => /lib/x86_64-linux-gnu/libresolv.so.2 (0x00007fb501176000)
libharfbuzz.so.0 => /usr/lib/x86_64-linux-gnu/libharfbuzz.so.0 (0x00007fb500f18000)
libthai.so.0 => /usr/lib/x86_64-linux-gnu/libthai.so.0 (0x00007fb500d0f000)
libexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007fb500ae6000)
libXt.so.6 => /usr/lib/x86_64-linux-gnu/libXt.so.6 (0x00007fb50087d000)
libudev.so.1 => /lib/x86_64-linux-gnu/libudev.so.1 (0x00007fb5380b2000)
libv4lconvert.so.0 => /usr/lib/x86_64-linux-gnu/libv4lconvert.so.0 (0x00007fb500603000)
libsoxr.so.0 => /usr/lib/x86_64-linux-gnu/libsoxr.so.0 (0x00007fb50039e000)
libnuma.so.1 => /usr/lib/x86_64-linux-gnu/libnuma.so.1 (0x00007fb500193000)
libogg.so.0 => /usr/lib/x86_64-linux-gnu/libogg.so.0 (0x00007fb4fff8a000)
liborc-0.4.so.0 => /usr/lib/x86_64-linux-gnu/liborc-0.4.so.0 (0x00007fb4ffd0a000)
libgcrypt.so.20 => /lib/x86_64-linux-gnu/libgcrypt.so.20 (0x00007fb4ffa29000)
libgssapi_krb5.so.2 => /usr/lib/x86_64-linux-gnu/libgssapi_krb5.so.2 (0x00007fb4ff7df000)
libhogweed.so.4 => /usr/lib/x86_64-linux-gnu/libhogweed.so.4 (0x00007fb4ff5ac000)
libnettle.so.6 => /usr/lib/x86_64-linux-gnu/libnettle.so.6 (0x00007fb4ff376000)
libgmp.so.10 => /usr/lib/x86_64-linux-gnu/libgmp.so.10 (0x00007fb4ff0f6000)
libxml2.so.2 => /usr/lib/x86_64-linux-gnu/libxml2.so.2 (0x00007fb4fed3b000)
libp11-kit.so.0 => /usr/lib/x86_64-linux-gnu/libp11-kit.so.0 (0x00007fb4fead7000)
libidn.so.11 => /usr/lib/x86_64-linux-gnu/libidn.so.11 (0x00007fb4fe8a4000)
libtasn1.so.6 => /usr/lib/x86_64-linux-gnu/libtasn1.so.6 (0x00007fb4fe691000)
libXau.so.6 => /usr/lib/x86_64-linux-gnu/libXau.so.6 (0x00007fb4fe48d000)
libXdmcp.so.6 => /usr/lib/x86_64-linux-gnu/libXdmcp.so.6 (0x00007fb4fe287000)
libgraphite2.so.3 => /usr/lib/x86_64-linux-gnu/libgraphite2.so.3 (0x00007fb4fe061000)
libdatrie.so.1 => /usr/lib/x86_64-linux-gnu/libdatrie.so.1 (0x00007fb4fde59000)
libSM.so.6 => /usr/lib/x86_64-linux-gnu/libSM.so.6 (0x00007fb4fdc51000)
libICE.so.6 => /usr/lib/x86_64-linux-gnu/libICE.so.6 (0x00007fb4fda37000)
libgpg-error.so.0 => /lib/x86_64-linux-gnu/libgpg-error.so.0 (0x00007fb4fd823000)
libkrb5.so.3 => /usr/lib/x86_64-linux-gnu/libkrb5.so.3 (0x00007fb4fd551000)
libk5crypto.so.3 => /usr/lib/x86_64-linux-gnu/libk5crypto.so.3 (0x00007fb4fd322000)
libcom_err.so.2 => /lib/x86_64-linux-gnu/libcom_err.so.2 (0x00007fb4fd11e000)
libkrb5support.so.0 => /usr/lib/x86_64-linux-gnu/libkrb5support.so.0 (0x00007fb4fcf13000)
libicuuc.so.55 => /usr/lib/x86_64-linux-gnu/libicuuc.so.55 (0x00007fb4fcb7f000)
libuuid.so.1 => /lib/x86_64-linux-gnu/libuuid.so.1 (0x00007fb4fc97a000)
libkeyutils.so.1 => /lib/x86_64-linux-gnu/libkeyutils.so.1 (0x00007fb4fc776000)
libicudata.so.55 => /usr/lib/x86_64-linux-gnu/libicudata.so.55 (0x00007fb4facbf000)
```

python3

```bash
linux-vdso.so.1 =>  (0x00007ffea3e50000)
libcaffe.so.1.0.0-rc5 => /tmp/code/quickdetection3.6/src/CCSCaffe/python/caffe/../../build/lib/libcaffe.so.1.0.0-rc5 (0x00007fa002273000)
libglog.so.0 => /usr/lib/x86_64-linux-gnu/libglog.so.0 (0x00007fa002044000)
libprotobuf.so.17 => /raid/jianfw/anaconda3/lib/libprotobuf.so.17 (0x00007fa001dcb000)
libboost_system.so.1.67.0 => /raid/jianfw/anaconda3/lib/libboost_system.so.1.67.0 (0x00007fa001bc6000)
libboost_filesystem.so.1.67.0 => /raid/jianfw/anaconda3/lib/libboost_filesystem.so.1.67.0 (0x00007fa0019aa000)
libboost_regex.so.1.67.0 => /raid/jianfw/anaconda3/lib/libboost_regex.so.1.67.0 (0x00007fa0016a6000)
libstdc++.so.6 => /raid/jianfw/anaconda3/lib/libstdc++.so.6 (0x00007fa003b69000)
libboost_python36.so.1.67.0 => /raid/jianfw/anaconda3/lib/libboost_python36.so.1.67.0 (0x00007fa001468000)
libpython3.6m.so.1.0 => /raid/jianfw/anaconda3/lib/libpython3.6m.so.1.0 (0x00007fa00111e000)
libgcc_s.so.1 => /raid/jianfw/anaconda3/lib/libgcc_s.so.1 (0x00007fa003b53000)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fa000d54000)
libcudart.so.9.0 => /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so.9.0 (0x00007fa000ae7000)
libcublas.so.9.0 => /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcublas.so.9.0 (0x00007f9ffd6b1000)
libcurand.so.9.0 => /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcurand.so.9.0 (0x00007f9ff974d000)
libgflags.so.2 => /usr/lib/x86_64-linux-gnu/libgflags.so.2 (0x00007f9ff952c000)
libhdf5_hl.so.100 => /raid/jianfw/anaconda3/lib/libhdf5_hl.so.100 (0x00007f9ff9307000)
libhdf5.so.101 => /raid/jianfw/anaconda3/lib/libhdf5.so.101 (0x00007f9ff8d74000)
libleveldb.so.1 => /usr/lib/x86_64-linux-gnu/libleveldb.so.1 (0x00007f9ff8b1a000)
libsnappy.so.1 => /raid/jianfw/anaconda3/lib/libsnappy.so.1 (0x00007f9ff8911000)
liblmdb.so.0 => /usr/lib/x86_64-linux-gnu/liblmdb.so.0 (0x00007f9ff86fc000)
libopencv_core.so.3.4 => /raid/jianfw/anaconda3/lib/libopencv_core.so.3.4 (0x00007f9ff7a65000)
libopencv_highgui.so.3.4 => /raid/jianfw/anaconda3/lib/libopencv_highgui.so.3.4 (0x00007fa003b40000)
libopencv_imgproc.so.3.4 => /raid/jianfw/anaconda3/lib/libopencv_imgproc.so.3.4 (0x00007f9ff5403000)
libopencv_imgcodecs.so.3.4 => /raid/jianfw/anaconda3/lib/libopencv_imgcodecs.so.3.4 (0x00007f9ff4ef8000)
libopencv_videoio.so.3.4 => /raid/jianfw/anaconda3/lib/libopencv_videoio.so.3.4 (0x00007fa003afe000)
libboost_thread.so.1.67.0 => /raid/jianfw/anaconda3/lib/libboost_thread.so.1.67.0 (0x00007f9ff4cd5000)
libcudnn.so.7 => /usr/lib/x86_64-linux-gnu/libcudnn.so.7 (0x00007f9fe25c3000)
libnccl.so.2 => /usr/lib/x86_64-linux-gnu/libnccl.so.2 (0x00007f9fdd8be000)
libmkl_rt.so => /raid/jianfw/anaconda3/lib/libmkl_rt.so (0x00007f9fdd1d8000)
libmpi.so.40 => /usr/local/lib/libmpi.so.40 (0x00007f9fdced9000)
libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f9fdcbd0000)
libgomp.so.1 => /raid/jianfw/anaconda3/lib/libgomp.so.1 (0x00007fa003ad6000)
libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f9fdc9b2000)
/lib64/ld-linux-x86-64.so.2 (0x00007fa003ab0000)
libunwind.so.8 => /usr/local/lib/libunwind.so.8 (0x00007f9fdc79a000)
libz.so.1 => /raid/jianfw/anaconda3/lib/./libz.so.1 (0x00007f9fdc582000)
librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f9fdc37a000)
libicudata.so.58 => /raid/jianfw/anaconda3/lib/./libicudata.so.58 (0x00007f9fda879000)
libicui18n.so.58 => /raid/jianfw/anaconda3/lib/./libicui18n.so.58 (0x00007f9fda404000)
libicuuc.so.58 => /raid/jianfw/anaconda3/lib/./libicuuc.so.58 (0x00007f9fda056000)
libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f9fd9e51000)
libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007f9fd9c4e000)
libjpeg.so.9 => /raid/jianfw/anaconda3/lib/./libjpeg.so.9 (0x00007f9fd9a11000)
libpng16.so.16 => /raid/jianfw/anaconda3/lib/./libpng16.so.16 (0x00007f9fd99d8000)
libtiff.so.5 => /raid/jianfw/anaconda3/lib/./libtiff.so.5 (0x00007f9fd995c000)
libjasper.so.4 => /raid/jianfw/anaconda3/lib/./libjasper.so.4 (0x00007f9fd96fd000)
libavcodec.so.58 => /raid/jianfw/anaconda3/lib/./libavcodec.so.58 (0x00007f9fd80af000)
libavformat.so.58 => /raid/jianfw/anaconda3/lib/./libavformat.so.58 (0x00007f9fd7c5a000)
libavutil.so.56 => /raid/jianfw/anaconda3/lib/./libavutil.so.56 (0x00007f9fd79eb000)
libswscale.so.5 => /raid/jianfw/anaconda3/lib/./libswscale.so.5 (0x00007f9fd7761000)
libopen-rte.so.40 => /usr/local/lib/libopen-rte.so.40 (0x00007f9fd74aa000)
libopen-pal.so.40 => /usr/local/lib/libopen-pal.so.40 (0x00007f9fd71a1000)
liblzma.so.5 => /raid/jianfw/anaconda3/lib/liblzma.so.5 (0x00007f9fd6f7b000)
libswresample.so.3 => /raid/jianfw/anaconda3/lib/././libswresample.so.3 (0x00007f9fd6d5d000)
libvpx.so.5 => /raid/jianfw/anaconda3/lib/././libvpx.so.5 (0x00007f9fd68dc000)
libopus.so.0 => /raid/jianfw/anaconda3/lib/././libopus.so.0 (0x00007f9fd6876000)
libbz2.so.1.0 => /raid/jianfw/anaconda3/lib/././libbz2.so.1.0 (0x00007f9fd6664000)
libnuma.so.1 => /usr/lib/x86_64-linux-gnu/libnuma.so.1 (0x00007f9fd6458000)
libudev.so.1 => /lib/x86_64-linux-gnu/libudev.so.1 (0x00007f9fd6438000)
```

