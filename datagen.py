import bpy, csv, cv2, colorsys, os, pathlib, random, sys, string, json, argparse, warnings, math
from types import SimpleNamespace
import numpy as np
import imutils, mathutils
from contextlib import redirect_stdout
from collections import defaultdict
from datetime import datetime


# How to use:
# blender --background --use-extension 1 -E CYCLES -t 0 -P 'dataset_coco_v2.py' -- --dir './dir' ...
# blender is your path to blender
# for macos it is /Applications/Blender.app/Contents/MacOS/blender

def clamp(x, minimum, maximum):
	return max(minimum, min(x, maximum))

def get_bbox(scene, cam_ob, me_ob):
	mat = cam_ob.matrix_world.normalized().inverted()
	depsgraph = bpy.context.evaluated_depsgraph_get()
	mesh_eval = me_ob.evaluated_get(depsgraph)
	me = mesh_eval.to_mesh()
	me.transform(me_ob.matrix_world)
	me.transform(mat)
	camera = cam_ob.data
	frame = [-v for v in camera.view_frame(scene=scene)[:3]]
	camera_persp = camera.type != 'ORTHO'
	lx = []
	ly = []
	for v in me.vertices:
		co_local = v.co
		z = -co_local.z
		if camera_persp:
			if z == 0.0:
				lx.append(0.5)
				ly.append(0.5)
			# Does it make any sense to drop these?
			# if z <= 0.0:
			# 	continue
			else:
				frame = [(v / (v.z / z)) for v in frame]
		min_x, max_x = frame[1].x, frame[2].x
		min_y, max_y = frame[0].y, frame[1].y
		x = (co_local.x - min_x) / (max_x - min_x)
		y = (co_local.y - min_y) / (max_y - min_y)
		lx.append(x)
		ly.append(y)
	min_x = clamp(min(lx), 0.0, 1.0)
	max_x = clamp(max(lx), 0.0, 1.0)
	min_y = clamp(min(ly), 0.0, 1.0)
	max_y = clamp(max(ly), 0.0, 1.0)
	mesh_eval.to_mesh_clear()
	r = scene.render
	fac = r.resolution_percentage * 0.01
	dim_x = r.resolution_x * fac
	dim_y = r.resolution_y * fac
	# Sanity check
	if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
		return (0, 0, 0, 0)
	return (
		round(min_x * dim_x),			 # X
		round(dim_y - max_y * dim_y),	 # Y
		round((max_x - min_x) * dim_x),  # Width
		round((max_y - min_y) * dim_y))   # Height)

def radians(degrees):
	return degrees * np.pi / 180

def objcolor():
	# object hsv
	o_h = random.uniform(0,1)
	o_s = 1.0
	o_v = random.uniform(0.7, 1.0)
	# alpha hsv
	a_h = random.uniform(0,1)
	a_s = 1.0
	a_v = 0.2
	scolor = tuple([*colorsys.hsv_to_rgb(o_h, o_s, o_v), 1.0])
	acolor = tuple([*colorsys.hsv_to_rgb(a_h, a_s, a_v), 1.0])
	return scolor, acolor

def get_material_name(obj):
	mesh = obj.data
	mat_name = ''
	for f in mesh.polygons:  # iterate over faces
		slot = obj.material_slots[f.material_index]
		mat = slot.material
		mat_name = mat.name
	return mat_name

class BlenderArgParse(argparse.ArgumentParser):
	def _get_argv_after_doubledash(self):
		try:
			idx = sys.argv.index("--")
			return sys.argv[idx+1:] # the list after '--'
		except ValueError as e: # '--' not in the list:
			return []
	# overrides superclass
	def parse_args(self):
		return super().parse_args(args=self._get_argv_after_doubledash())

# only consider args after '--', because blender needs its own args
argv = sys.argv
argv = argv[argv.index('--') + 1:]
parser = BlenderArgParse()
parser.add_argument(
	'--dir',
	type = str,
	help = 'The directory in which to place the dataset.',
	required = True
)
parser.add_argument(
	'--bg_dir',
	type = str,
	help = 'The directory containing background images.',
	required = True
)
parser.add_argument(
	'--shape_dir',
	type = str,
	default = './shapes',
	help = 'The directory containing shape .obj and .mtl files. A shape\'s .obj must contain a .mtl with the material parameters of the shape with the same name.',
	required = True
)
parser.add_argument(
	'--alpha_dir',
	type = str,
	default = './alphas',
	help = 'The directory containing alphanumeric .obj and .mtl files. An alphanumeric\'s .obj must contain a corresponding .mtl with the same filename.',
	required = True
)
parser.add_argument(
	'--n',
	type = int,
	help = 'no. of images to generate per shape',
	required = True
)
parser.add_argument(
	'--wl',
	type = float,
	default = 0.03,
	help = 'shape minimum width, in meters.',
	required = False
)
parser.add_argument(
	'--wu',
	type = float,
	default = 0.08,
	help = 'shape maximum width, in meters.',
	required = False
)
parser.add_argument(
	'--res',
	type = int,
	default = 512,
	help = 'The image width/height (generated images are square)',
	required = False
)
parser.add_argument(
	'--pu', 
	type = float, 
	default = 60, 
	help = 'upper camera pitch range (degrees). 90 = horizon',
	required = False
)
parser.add_argument(
	'--pl',
	type = float,
	default = 0,
	help = 'lower camera pitch (degrees). 0 = straight at ground',
	required = False
)
parser.add_argument(
	'--cam_roll', 
	type = float,
	default = 20,
	help = 'Camera roll, in degrees. The camera will be rolled with a uniform distribution over images in the range [-beta, +beta], where beta is the argument given.',
	required = False
)
parser.add_argument(
	'--cam_focal_len',
	type = int,
	default = 35,
	help = 'Camera focal length, in mm.',
	required = False
)
parser.add_argument(
	'--al',
	type = float,
	default = .2, 
	help = 'Camera altitude, lower. Meters',
	required = False
)
parser.add_argument(
	'--au',
	type = float,
	default = .8, 
	help = 'Camera altitude, upper. Meters',
	required = False
)
parser.add_argument(
	'--norender',
	type = bool,
	default = False,
	help = 'Don\'t actually render the images',
	required = False
)
parser.add_argument(
	'--alphascale',
	type = float,
	default = 0.4,
	help = 'widest point of letter/char compared to widest point of shape',
	required = False
)

opt = parser.parse_args()

console = True

# create dirs
dirs = SimpleNamespace()
dirs.dsetdir = opt.dir
dirs.imagedir = opt.dir + '/images'
dirs.labeldir = opt.dir + '/labels'
dirs.masksdir = opt.dir + '/masks'
dirs.bg_paths = [b.path for b in os.scandir(opt.bg_dir) if b.is_file()]

os.makedirs(dirs.imagedir)
os.makedirs(dirs.labeldir)
os.makedirs(dirs.masksdir)

# get shapes and alphas
shape_list = sorted([s[:-4] for s in os.listdir(opt.shape_dir) if s[-4:] != '.mtl'])
alpha_list = sorted([a[:-4] for a in os.listdir(opt.alpha_dir) if a[-4:] != '.mtl'])
padzeros = math.ceil(math.log10(len(shape_list) * len(alpha_list) * opt.n))

# annotations 
anno = SimpleNamespace()
anno.categories = {}
anno.images, anno.annos = [], []

# for every shape, sometimes we need to offset the letter within it,
# ie we cannot center the letter perfectly (like how a triangle's centroid
# is "off" from its bbox center if standing up.
alpha_zoff = 0.0005
shape_offsets = defaultdict(
	lambda: [0,0,0],
	{
	'Circle': [0, 0, 0],
	'Diamond': [0, 0, 0],
	'Halfcircle': [0.00, 0, 0],
	'Heart': [0, 0, 0],
	'Hexagon': [0, 0, 0],
	'Hexstar': [0, 0, 0],
	'Moon': [-0.10, 0, 0],
	'Octagon': [0, 0, 0],
	'Pentagon': [0, 0, 0],
	'Pentstar': [0, 0, 0],
	'Plus': [0, 0, 0],
	'Square': [0, 0, 0],
	'Trapezoid': [0, 0, 0],
	'Triangle': [0, -0.1, 0],
})
shape_scales = defaultdict(
	lambda: 1,
	{
	'Circle': 1,
	'Diamond': .5,
	'Halfcircle': .65,
	'Heart': 0.9,
	'Hexagon': 1,
	'Hexstar': 0.8,
	'Moon': .6,
	'Octagon': 1,
	'Pentagon': 1,
	'Pentstar': .6,
	'Plus': .8,
	'Square': 1,
	'Trapezoid': .7, 
	'Triangle': .7,}
)

bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'

startt = datetime.now()
n = 0
for alpha_i, alpha in enumerate(alpha_list):
	for shape_i, shape in enumerate(shape_list):
		if not shape in anno.categories:
			anno.categories[shape] = {
				'supercategory' : 'shape',
				'id' : shape_i,
				'name' : shape
			}
		m = 0
			
		while m < opt.n:
			# collect paths into namespace
			paths = SimpleNamespace()

			# reset scene to empty
			bpy.ops.wm.read_factory_settings(use_empty=True)
			if console:
				print('reset scene.')
				print('working on image ' + str(n).zfill(padzeros) + ':')

			# define save paths
			obj_path = str(n).zfill(padzeros) + '__' + alpha + '_' + shape
			paths.image = (dirs.imagedir + '/', str(obj_path) + '.jpg')
			paths.mask = (dirs.masksdir + '/', str(obj_path) + '.png')
			if console:
				print('\tsave image to: ',*paths.image, sep='')
				print('\tsave mask to: ',*paths.mask, sep='')
			
			# get object paths
			paths.shape = opt.shape_dir + '/' + shape + '.obj'
			paths.shapemtl = opt.shape_dir + '/' + shape + '.mtl'
			paths.alpha = opt.alpha_dir + alpha + '.obj'
			paths.alphamtl = opt.alpha_dir + alpha + '.mtl'
			if console:
				print('\tget shape from: {}'.format(paths.shape))
				print('\tget alpha from: {}'.format(paths.alpha))

			# import scene
			with redirect_stdout(None):
				bpy.ops.import_scene.obj(filepath = paths.shape)
				bpy.ops.import_scene.obj(filepath = paths.alpha)

			# render settings
			bpy.data.scenes['Scene'].render.resolution_x = opt.res
			bpy.data.scenes['Scene'].render.resolution_y = opt.res
			bpy.context.scene.render.image_settings.file_format='JPEG'

			# lighting
			sun_loc = mathutils.Vector([np.random.normal(0, 1000), np.random.normal(0, 1000), 1000])
			bpy.ops.object.light_add(type='SUN', location=sun_loc)

			# set origin as object origin for letter, shape in scene.
			bpy.ops.object.select_all(action='DESELECT')
			bpy.data.objects[shape].select_set(True)
			bpy.data.objects[alpha].select_set(True)
			bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
			dg = bpy.context.evaluated_depsgraph_get() 
			bpy.data.objects[shape].rotation_mode = 'XYZ' 
			bpy.ops.object.select_all(action='DESELECT')

			# background TODO: refactor
			bgpath = random.choice(dirs.bg_paths)
			bpy.context.scene.render.film_transparent = True
			bpy.context.scene.use_nodes = True
			composite = bpy.context.scene.node_tree.nodes[0]
			render_layers = bpy.context.scene.node_tree.nodes[1]
			alpha_over = bpy.context.scene.node_tree.nodes.new(type = 'CompositorNodeAlphaOver')
			background_img_node = bpy.context.scene.node_tree.nodes.new(type = 'CompositorNodeImage')
			scale_node = bpy.context.scene.node_tree.nodes.new(type = 'CompositorNodeScale')
			links = bpy.context.scene.node_tree.links
			link_1 = links.new(render_layers.outputs[0], alpha_over.inputs[2])
			link_2 = links.new(alpha_over.outputs[0], composite.inputs[0])
			link_3 = links.new(background_img_node.outputs[0], scale_node.inputs[0])
			link_4 = links.new(scale_node.outputs[0], alpha_over.inputs[1])
			bpy.data.scenes['Scene'].node_tree.nodes['Scale'].space = 'RENDER_SIZE'

			# shape color
			scolor, acolor = objcolor()
			if console:
				print('\tshape is color: {}'.format([round(c, 2) for c in scolor]))
				print('\talpha is color: {}'.format([round(c, 2) for c in acolor]))
			matname = get_material_name(bpy.data.objects[shape])
			bpy.data.materials[matname].node_tree.nodes['Principled BSDF'].inputs[7].default_value = .5
			bpy.data.materials[matname].node_tree.nodes['Principled BSDF'].inputs[0].default_value = scolor
			matname = get_material_name(bpy.data.objects[alpha])
			bpy.data.materials[matname].node_tree.nodes['Principled BSDF'].inputs[7].default_value = .7
			bpy.data.materials[matname].node_tree.nodes['Principled BSDF'].inputs[0].default_value = acolor
			
			# scale shape/alpha
			shapew = random.uniform(opt.wl, opt.wu)
			# scale shape
			shape_scalefactor = shapew / max(list(bpy.data.objects[shape].dimensions)) 
			bpy.data.objects[shape].select_set(True)
			bpy.data.objects[alpha].select_set(False)
			bpy.ops.transform.resize(
				value = (shape_scalefactor, shape_scalefactor, shape_scalefactor),
				orient_type = 'LOCAL'
			)
			bpy.ops.object.select_all(action='DESELECT')
			# scale alpha
			alpha_scalefactor = opt.alphascale * max(list(bpy.data.objects[shape].dimensions)) / max(list(bpy.data.objects[alpha].dimensions)) * shape_scales[shape]
			bpy.data.objects[shape].select_set(False)
			bpy.data.objects[alpha].select_set(True)
			bpy.ops.transform.resize(
				value = (alpha_scalefactor, alpha_scalefactor, alpha_scalefactor),
				orient_type = 'LOCAL'
			)
			# scale the offset to the shape
			letteroffset = mathutils.Vector(shape_offsets[shape]) * max(list(bpy.data.objects[shape].dimensions))

			# camera TODO: put this into function
			psi = random.uniform(opt.pl, opt.pu)
			beta = random.uniform(-opt.cam_roll, opt.cam_roll)
			dist = random.uniform(opt.al, opt.au)
			if console:
				print('\tadd camera: altitude={}'.format(round(dist,2)))
				print('\trotate camera: pitch={}, roll={}'.format(round(psi,2), round(beta,2)))
			
			# add camera
			bpy.ops.object.camera_add(location=(0, 0, dist))
		
			if console:
				print('\tobjects in scene: ')
				for o in bpy.context.scene.objects:
					print('\t\t{}'.format(o))

			cam = bpy.context.object
			# random camera rotation
			cam.data.clip_end = 1e+08
			cam.data.type = 'PERSP'
			cam.data.lens_unit = 'MILLIMETERS'
			cam.data.lens = opt.cam_focal_len
			bpy.context.scene.camera = cam
			# select and rotate camera
			for obj in bpy.context.selected_objects:
				obj.select_set(False)
			# select camera
			for obj in bpy.context.scene.objects:
				if obj.type == 'CAMERA':
					obj.select_set(True)
			# randomly rotate camera
			bpy.ops.transform.rotate(value=radians(psi), orient_axis='X')
			bpy.ops.transform.rotate(value=radians(beta), orient_axis='Y')
			# deselect camera
			for obj in bpy.context.scene.objects:
				if obj.type == 'CAMERA':
					obj.select_set(False)
			# plane point
			plane_co = (0,0,0)
			# plane normal
			plane_no = (0,0,1)
			# camera worldspace transformation matrix
			mw = cam.matrix_world
			# camera worldspace translation
			o = mw.translation
			# top right, bottom right, etc. for each point defining the camera frame
			tr, br, bl, tl = [mw @ f for f in cam.data.view_frame(scene=bpy.context.scene)]
			# length of camera frame on the ground, x
			x = tr - tl
			# length of camera frame on the ground, y
			y = tr - br
			# place an object randomly on CAMERA view
			#	Camera View:
			#	┌──────────────────┐
			#	│                  │
			#	│                  │
			#	│  . <cx, cy>      │
			#	│                  │
			#	└──────────────────┘
			cx, cy = random.uniform(0.1, 0.9), random.uniform(0.1, 0.8)
			# vector pointing from camera origin through that <cx, cy> point
			v = (bl + (cx * x + cy * y)) - o
			# point's location where that vector intersects origin xy plane
			pt = mathutils.geometry.intersect_line_plane(o, o + v, plane_co, plane_no, True)
			
			# check
			if pt and (pt - o).dot(v) > 0:
				if console:
					print('\tadding object at: {}'.format(pt))
				else:
					pass
			else:
				raise(Exception('Point added violates camera bounds. Check that camera is pointing in -z direction.'))

			# randomly rotate letter: 0, 90 deg, 180 deg, 270 deg.
			letter_rots = [0, np.pi/2, np.pi, 3*np.pi/2]
			bpy.data.objects[alpha].select_set(True)
			bpy.ops.transform.rotate(value=random.choice(letter_rots))
			bpy.data.objects[alpha].select_set(False)

			# place shape
			bpy.data.objects[shape].location = pt
			bpy.data.objects[alpha].location = pt \
				+ mathutils.Vector([0,0,0.0001]) + letteroffset

			# set shape as parent to letter
			bpy.data.objects[alpha].select_set(True)
			bpy.data.objects[shape].select_set(True)
			bpy.context.view_layer.objects.active = bpy.data.objects[shape]
			bpy.ops.object.parent_set()
			bpy.data.objects[alpha].select_set(False)
			bpy.ops.transform.rotate(value=random.uniform(0,2*np.pi), orient_axis='Z')

			if console:
				print("\tshape dims = {}, alpha dims = {}".format(
					bpy.data.objects[shape].dimensions, 
					bpy.data.objects[alpha].dimensions))
			bpy.data.objects[shape].select_set(False)

			x, y, w, h = get_bbox(bpy.context.scene, bpy.context.scene.camera, bpy.data.objects[shape])
			
			# save bbox anno
			xcent, ycent = (x + (w/2))/opt.res, (y + (h/2))/opt.res
			normwidth, normheight = (w/opt.res, h/opt.res)
			row = [str(shape_i), str(xcent), str(ycent), str(normwidth), str(normheight)]
			rowj = ' '.join(row)
			textfname = dirs.labeldir + '/' + obj_path + '.txt', 'w'
			with open(*textfname) as txtf:
				txtf.write(rowj)



			# annotations
			image_info = {
				'id' : n,
				'height' : opt.res,
				'width' : opt.res,
				'file_name' : ''.join(paths.image),
				'license': 1
			}
			anno.images.append(image_info)

			anno_info = {
				'id' : (shape_i) * len(alpha_list) + m,
				'image_id' : n,
				'category_id' : shape_i,
				# 'segmentation' : poly,
				# 'area' : poly_a,
				'bbox' : [x, y, w, h],
				'iscrowd' : 0,
				# 'mask_file' : mask_name[1],
			}
			anno.annos.append(anno_info)

			# background
			background_img_node.image = bpy.data.images.load(bgpath)
			bpy.data.scenes['Scene'].render.filepath = ''.join(paths.image)
			if not opt.norender:
				with redirect_stdout(None):
					bpy.ops.render.render(write_still=True)
			m += 1
			n += 1



print('Categories:')
for k, v in anno.categories.items():
	print(k)
	for k2, v2 in v.items():
		print('\t', k2, v2)

# fill categories
dset_categories = []
for shape, category in anno.categories.items():
	dset_categories.append(category)

# dsetinformation
dset_info = {
	'year' : 2021,
	'version' : '1.1.0',
	'description' : 'Target ID dataset for the 2021 UAV Forge SUAS Competition. {} target classes, {} images, resolution: {}x{}'.format(len(dset_categories), n, opt.res, opt.res),
	'contributor' : 'Mike Sutherland (msutherl@uci.edu)',
	'date_created' : str(datetime.now())
}

# licenses
dset_lic = [{
	'id' : 1,
	'name' : 'BY-NC',
	'url' : 'https://en.wikipedia.org/wiki/Creative_Commons_license'
}]

# annotations
dset = {
	'info' : dset_info,
	'categories' : dset_categories,
	'licenses' : dset_lic,
	'images' : anno.images,
	'annotations' : anno.annos,
}

dfile = opt.dir + 'annotation.json'
print('finished in {}. Writing JSON to {}...'.format(datetime.now() - startt, dfile))
with open(dfile, 'x') as anno_file:
	json.dump(dset, anno_file)
print('Done!')