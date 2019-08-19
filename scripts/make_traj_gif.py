import argparse
import path_utils
import plot_tools

parser = argparse.ArgumentParser()
parser.add_argument('traj_file')
parser.add_argument('--length', type=float, default='5.0')
args = parser.parse_args()

print('Making gif of trajectory from ', args.traj_file)

plot_tools.make_traj_gif(args.traj_file, length=args.length)



















#
