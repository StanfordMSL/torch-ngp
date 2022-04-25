### Translation study

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/trans/1em3/nopt --bound 1.5 --scale 1.0 --mode blender --trans_noise 1e-3

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/trans/1em3/opt --bound 1.5 --scale 1.0 --mode blender --trans_noise 1e-3 --opt_poses

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/trans/1em2/nopt --bound 1.5 --scale 1.0 --mode blender --trans_noise 1e-2

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/trans/1em2/opt --bound 1.5 --scale 1.0 --mode blender --trans_noise 1e-2 --opt_poses

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/trans/1em1/nopt --bound 1.5 --scale 1.0 --mode blender --trans_noise 1e-1

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/trans/1em1/opt --bound 1.5 --scale 1.0 --mode blender --trans_noise 1e-1 --opt_poses

### Rotation study

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/rot/1em3/nopt --bound 1.5 --scale 1.0 --mode blender --rot_noise 1e-3

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/rot/1em3/opt --bound 1.5 --scale 1.0 --mode blender --rot_noise 1e-3 --opt_poses

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/rot/1em2/nopt --bound 1.5 --scale 1.0 --mode blender --rot_noise 1e-2

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/rot/1em2/opt --bound 1.5 --scale 1.0 --mode blender --rot_noise 1e-2 --opt_poses

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/rot/1em1/nopt --bound 1.5 --scale 1.0 --mode blender --rot_noise 1e-1

python main_nerf.py data/nerf_synthetic/lego --workspace logs/pose_study/rot/1em1/opt --bound 1.5 --scale 1.0 --mode blender --rot_noise 1e-1 --opt_poses

## Est time: 15 min x 6 + 25 min x 6 = 4 hours

