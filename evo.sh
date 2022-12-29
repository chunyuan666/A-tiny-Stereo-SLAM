source ~/.bashrc
evo_traj kitti trajectory.txt KITTI_00_ORB.txt --ref=KITTI_00_gt.txt -p --plot_mode=xz
# evo_ape kitti KITTI_00_gt.txt trajectory.txt -va --plot --plot_mode xz