clear
mkdir build
cd build
cmake ..
make -j8
cd ..
./bin/stereo_kitti /home/uav/A-tiny-Stereo-SLAM/config/KITTI00-02.yaml /home/uav/KITTI/02
evo_traj kitti trajectory.txt --ref=/home/uav/KITTI/data_odometry_poses/dataset/poses/02.txt -p --plot_mode=xz
