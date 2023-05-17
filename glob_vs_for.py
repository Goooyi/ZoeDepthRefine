import os
import glob
import time

from pathlib import Path


data_dir = f'/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_5scenes_newPitch_original'

# using os + for loop
start = time.time()

scene_names = sorted(os.listdir(os.path.join(data_dir, 'habitat_sim_DAVIS/JPEGImages/480p')))

for_loop_data = []

for scene_name in scene_names:
	rgb_folder = os.path.join(data_dir, 'habitat_sim_DAVIS/JPEGImages/480p',scene_name)
	depth_folder = os.path.join(data_dir, 'habitat_sim_DAVIS/Annotations/480p_depth', scene_name)
	mask_folder = os.path.join(data_dir, 'habitat_sim_DAVIS/Annotations/480p_objectID', scene_name)
	pseudo_depth_folder = os.path.join(data_dir, 'zoe_depth_raw', scene_name)
	
	for filename in os.listdir(rgb_folder):
		if filename.endswith('.png') or filename.endswith('.jpg'):
			rgb_path = os.path.join(rgb_folder, filename)
			depth_path = os.path.join(depth_folder, filename[:-4]+'.png')
			pseudo_depth_path = os.path.join(pseudo_depth_folder, filename[:-4]+'.png')
			mask_path = os.path.join(mask_folder, filename[:-4]+'.png')
			for_loop_data.append((rgb_path, depth_path, pseudo_depth_path, mask_path))
			
print(f"for loop result len: {len(for_loop_data)}")
print(for_loop_data[0])
end = time.time()
print(end-start)

# using glob
glob_data = []

rgb_folder = os.path.join(data_dir, 'habitat_sim_DAVIS/JPEGImages/480p')
depth_folder = os.path.join(data_dir, 'habitat_sim_DAVIS/Annotations/480p_depth')
mask_folder = os.path.join(data_dir, 'habitat_sim_DAVIS/Annotations/480p_objectID')
pseudo_depth_folder = os.path.join(data_dir, 'zoe_depth_raw', scene_name)

rgb_path = glob.glob(r"*/*.jpg", root_dir=rgb_folder)
depth_path = glob.glob('*/*.png', root_dir=depth_folder)
mask_path = glob.glob('*/*.png', root_dir=mask_folder)
pseudo_depth_path = glob.glob('**/*.png', root_dir=depth_folder)
glob_data = list(zip(rgb_path, depth_path, mask_path, pseudo_depth_path))

print(f"glob len: {len(for_loop_data)}")
print(glob_data[0])
end = time.time()
print(end-start)