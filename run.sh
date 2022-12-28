clear
mkdir build
cd build
cmake ..
make -j8
cd ..
./bin/stereo_kitti /home/uav/A-tiny-Stereo-SLAM/config/KITTI00-02.yaml /home/uav/KITTI/00
