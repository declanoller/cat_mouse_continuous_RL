
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import path_utils
import os, subprocess, json, shutil, glob



def save_traj_to_file(traj, dir, **kwargs):

	'''

	For saving trajectories. Used by the RL algos.

	Just using jsons here because it's easier to pass other info.

	'''


	traj_dir = os.path.join(dir, 'traj_plots')

	if not os.path.exists(traj_dir):
		os.mkdir(traj_dir)

	if kwargs.get('iter', None) is not None:
		fname = os.path.join(traj_dir, 'traj_{}.json'.format(kwargs.get('iter')))
	else:
		fname = os.path.join(traj_dir, 'traj.json')


	traj_dict = {
		'mouse_pos_x' : traj[:,0].tolist(),
		'mouse_pos_y' : traj[:,1].tolist(),
		'cat_pos_x' : traj[:,2].tolist(),
		'cat_pos_y' : traj[:,3].tolist(),
		'cat_speed_rel' : kwargs.get('cat_speed_rel', None),
		'mouse_caught' : kwargs.get('mouse_caught', None)
	}
	# use 0 and 1 for mouse_caught because json doesnt play nice with python bools.

	with open(fname, 'w+') as f:
		json.dump(traj_dict, f, indent=4)

	return fname





def plot_traj_from_file(traj_file, **kwargs):

	'''
	Plots a catmouse trajectory from file. Expects a numpy
	file that was saved with np.savetxt, having 4 columns: mouse pos, cat pos.

	Just a load wrapper for plot_traj.

	'''

	# Create a pics dir in the same dir as the file, if one isn't created yet
	dir = os.path.dirname(traj_file)
	traj_pics_dir = os.path.join(dir, 'traj_plots')

	if not os.path.exists(traj_pics_dir):
		os.mkdir(traj_pics_dir)

	with open(traj_file, 'r') as f:
		traj_dict = json.load(f)

	traj = np.stack(
						traj_dict['mouse_pos_x'],
						traj_dict['mouse_pos_y'],
						traj_dict['cat_pos_x'],
						traj_dict['cat_pos_y'],
						axis=1
					)


	traj_basename = os.path.basename(traj_file)[0]
	traj_pic_fname = os.path.join(traj_pics_dir, f'{traj_basename}.png')

	plot_traj(traj, traj_pic_fname,
					cat_speed_rel=traj_dict['cat_speed_rel'],
					mouse_caught=traj_dict['mouse_caught'])



def plot_traj(traj_array, output_fname, **kwargs):

	'''
	Plots a passed trajectory (numpy array) with 4 columns: mouse pos, cat pos.

	'''

	im_mouse = Image.open(os.path.join(path_utils.get_misc_dir(), 'mouse_transparent_small.png'))
	#im_cat = Image.open(os.path.join(path_utils.get_misc_dir(), 'cat_transparent_small.png'))
	im_cat = Image.open(os.path.join(path_utils.get_misc_dir(), 'cat_transparent_small_with_margin.png'))

	# In axes units, don't need to worry about pixels or anything
	mouse_w = 0.1
	mouse_h = 0.1

	cat_h = 0.4
	cat_w = 1.05*cat_h

	plt.close('all')
	s = 6
	fig = plt.figure(figsize=(s,s), frameon=False)

	ax = plt.gca()
	ax.axis('off')
	plt.autoscale(tight=True)

	mouse_pos = traj_array[:,:2]
	# Put can slightly outside main circle
	cat_rad = 1.0 + 0.55*cat_h
	cat_pos = cat_rad*traj_array[:,2:]

	cat_pos_x = cat_pos[:,0]
	cat_pos_y = cat_pos[:,1]
	#cat_pos_x = cat_rad*np.cos(cat_angle)
	#cat_pos_y = cat_rad*np.sin(cat_angle)
	#print(mouse_pos)
	plt.plot(mouse_pos[:,0], mouse_pos[:,1], color='dodgerblue')

	#circle_last_mouse_pos = plt.Circle((mouse_pos[-1,0], mouse_pos[-1,1]), 0.03, fill=False, edgecolor='mediumseagreen', linewidth=2)
	#ax.add_artist(circle_last_mouse_pos)

	# Draw mouse
	ax.imshow(im_mouse,
			extent=(mouse_pos[-1,0] - 1.3*mouse_w, mouse_pos[-1,0] + 0.7*mouse_w, mouse_pos[-1,1] - mouse_h, mouse_pos[-1,1] + mouse_h),
			zorder=10)
	#plt.scatter(mouse_pos[-1,0], mouse_pos[-1,1], s=60, edgecolor='black', linewidth=2.5, marker='o', facecolors='black')
	#print(mouse_pos[-1,0], mouse_pos[-1,1])

	# All positions
	plt.scatter(cat_pos_x, cat_pos_y, s=200, edgecolor='black', linewidth=0.2, marker='o', facecolors='None')
	# End and start positions
	plt.scatter(cat_pos_x[-1], cat_pos_y[-1], s=120, edgecolor='red', linewidth=2.5, marker='o', facecolors='None')
	plt.scatter(cat_pos_x[0], cat_pos_y[0], s=120, edgecolor='blue', linewidth=2.5, marker='o', facecolors='None')

	#plt.plot(cat_pos_x, cat_pos_y, color='black', markersize=8, marker='o', fillstyle=None)

	# Figure out the angle we need to rotate the cat to
	cat_last_ang = np.arctan2(cat_pos_y[-1], cat_pos_x[-1])
	cat_ang_degree = (180/3.14)*cat_last_ang
	#print(cat_last_ang)
	im_cat = im_cat.rotate(cat_ang_degree - 90)

	ax.imshow(im_cat,
			extent=(cat_pos[-1,0] - cat_w, cat_pos[-1,0] + cat_w, cat_pos[-1,1] - cat_h, cat_pos[-1,1] + cat_h),
			zorder=10)


	# Main outer circle
	circle_main = plt.Circle((0, 0), 1.0, edgecolor='tomato', fill=False)
	ax.add_artist(circle_main)

	# Circle the mouse can dash from
	cat_speed_rel = kwargs.get('cat_speed_rel', None)
	if cat_speed_rel is not None:
		r_dash = max(0, 1.0-3.14/cat_speed_rel)
		circle_dash = plt.Circle((0, 0), r_dash, linestyle='dashed', edgecolor='grey', fill=False)
		ax.add_artist(circle_dash)

		# Circle the mouse can match the cat's angular speed at
		r_match = 1.0/cat_speed_rel
		circle_match = plt.Circle((0, 0), r_match, edgecolor='lightgray', fill=False)
		ax.add_artist(circle_match)


	lim = 1.5
	ax.set_xlim(-lim, lim)
	ax.set_ylim(-lim, lim)
	ax.set_aspect('equal')

	# Add title if mouse escaped or was caught
	mouse_caught = kwargs.get('mouse_caught', None)
	if mouse_caught is not None:
		if mouse_caught == 'caught':
			plt.title('Caught!', fontsize=20)
		elif mouse_caught == 'escaped':
			plt.title('Escaped!', fontsize=20)

	#plt.tight_layout()
	# makes it tight right now, but no space for the title...
	plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
	#plt.show()
	plt.savefig(output_fname)






def make_traj_gif(traj_file, **kwargs):

	'''
	Repeatedly calls plot_traj to plot the traj incrementally, to make a gif.
	Then, creates a gif from the images produced.


	'''

	dir_name = os.path.dirname(traj_file)
	base_name = os.path.basename(traj_file)
	base_name = os.path.splitext(base_name)[0] # get rid of .txt
	traj_gif_dir = os.path.join(dir_name, f'gif_pics_{base_name}')
	if os.path.exists(traj_gif_dir):
		shutil.rmtree(traj_gif_dir)
	os.mkdir(traj_gif_dir)


	with open(traj_file, 'r') as f:
		traj_dict = json.load(f)

	# Get whole trajectory
	full_traj = np.stack(
						(traj_dict['mouse_pos_x'],
						traj_dict['mouse_pos_y'],
						traj_dict['cat_pos_x'],
						traj_dict['cat_pos_y']), axis=1)

	# Iterate through the whole trajectory
	for i in range(1, len(full_traj)+1):

		partial_traj = full_traj[:i]
		fname = os.path.join(traj_gif_dir, f'{i}.png')

		if i == len(full_traj):
			plot_traj(partial_traj, fname,
							cat_speed_rel=traj_dict['cat_speed_rel'],
							mouse_caught=traj_dict['mouse_caught'])
		else:
			plot_traj(partial_traj, fname,
							cat_speed_rel=traj_dict['cat_speed_rel'])

		if kwargs.get('sustain_last_frame', True):
			# Duplicates the last frame 5 times, makes a better gif
			# would probably be better
			for j in range(1, 5):
				copy_fname = os.path.join(traj_gif_dir, f'{i+j}.png')
				shutil.copyfile(fname, copy_fname)


	gif_fname = os.path.join(dir_name, f'{base_name}_traj.gif')
	# make gif
	imgs_to_gif(traj_gif_dir, gif_fname, **kwargs)

	print('Removing gif pics dir...')
	# Remove gif pics dir
	shutil.rmtree(traj_gif_dir)
	print('Done.')






def imgs_to_gif(imgs_dir, output_fname, **kwargs):

	ext = '.png'

	length = 1000.0*kwargs.get('length', 5.0) # passed in seconds, turned into ms

	file_list = glob.glob(os.path.join(imgs_dir, f'*{ext}')) # Get all the pngs in the current directory
	N_files = len(file_list)
	delay = length/N_files
	print(f'Making gif from {N_files} pics, using delay of {delay} for total length of {length}')
	#print(file_list)
	list.sort(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
	#print(file_list)
	assert len(file_list) < 300, 'Too many files ({}), will probably crash convert command.'.format(len(file_list))

	check_call_arglist = ['convert'] + ['-delay', str(delay)] + file_list + [output_fname]
	#print(check_call_arglist)
	print('Calling convert command to create gif...')
	subprocess.check_call(check_call_arglist)
	print('done.')
	return output_fname


#
