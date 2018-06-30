import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
import os
import time
# import seaborn
import numpy as np
import scipy as sc
# from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, delaunay_plot_2d
# from shapely.ops import polygonize
# from shapely.geometry import LineString, Polygon, MultiPolygon, MultiPoint, Point
# from descartes.patch import PolygonPatch
# from noise import snoise2
# import lz4
from concurrent import futures

from pyDome.Polyhedral import *
from pyDome.SymmetryTriangle import *
from pyDome.GeodesicSphere import *

# ##### pip install --no-binary protobuf
from world_gen_pb2_grpc import *
from world_gen_pb2 import *

from grpc._cython import cygrpc

cuda_block_size = 32
cuda_block_2d = (cuda_block_size, cuda_block_size, 1)

def make_stone(make_stone_cuda):
    """Generates a stone by deforming a geosphere"""
    import pycuda.autoinit
    import pycuda.driver as drv

    radius = np.float64(1.)
    frequency = 4
    polyhedral = Icosahedron()
    vertex_equal_threshold = 0.0000001
    truncation_amount = 0.499999
    symmetry_triangle = ClassOneMethodOneSymmetryTriangle(frequency, polyhedral)
    sphere = GeodesicSphere(polyhedral, symmetry_triangle, vertex_equal_threshold, radius)
    F_sphere = sphere.non_duplicate_face_nodes
    V_sphere = sphere.sphere_vertices
    orig_verts = np.array(V_sphere, dtype=np.float32)
    faces = np.array(F_sphere, dtype=np.int32) - 1
    nverts = orig_verts.shape[0]

    scale = 1.0 + np.abs(np.clip(np.random.randn(3), -3, 3)) * 0.5

    verts = np.zeros_like(orig_verts, dtype=np.float32)
    uvs = np.zeros((orig_verts.shape[0], 2), dtype=np.float32)
    block_size = 512
    make_stone_cuda(
            drv.Out(verts), drv.Out(uvs), drv.In(orig_verts), np.uint32(nverts), scale.astype(np.float32), np.int32(np.random.randint(0, 1000000)),
            block=(block_size, 1, 1), grid=(nverts // block_size + 1, 1)
    )

    return verts, uvs, faces

def gen_map(map_size=(256,256), map_scale=1.0, real_size=(500.0, 100.0, 500.0)):
    """Generates a map, with biomes, vegetation and some props (stones)"""
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    BIOME_WATER_DEEP = 0
    BIOME_WATER_SHALLOW = 1
    BIOME_GRASS = 2
    BIOME_ROCKS = 3
    BIOME_SNOW = 4
    BIOME_DIRT = 5

    map_size_total = ( map_size[0] + cuda_block_size, map_size[1] + cuda_block_size )
    cuda_grid_shape = (map_size_total[0] // cuda_block_size, map_size_total[1] // cuda_block_size)

    # Terrain mesh will be colored by biome
    biome_colors = np.array([
        to_rgb('#002244'), # Deep Water
        to_rgb('#2288aa'), # Water
        to_rgb('#228844'), # Forest
        to_rgb('#6c6964'), # Rocks
        to_rgb('#ffffff'), # Mountains
        to_rgb('#4c4944') # Dirt
    ])
    nbiomes = biome_colors.shape[0]
    # Load CUDA-based generation functions
    with open('cuda/terrain_base.cu', 'r') as f:
        mod = SourceModule(f.read() % {
                'map_w': map_size_total[0],
                'map_h': map_size_total[1],
                'map_scale': map_scale * 4000.0,
                'slope_scale': 1.0 / (real_size[0] / map_size[0]) * real_size[1],
                'nbiomes': nbiomes
            },
            include_dirs=[ os.getcwd() ],
            no_extern_c=True,
            options=[ '-Xcudafe', '"--diag_suppress=code_is_unreachable,branch_past_initialization"' ]
        )
    # warning suppression codes: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
    make_noise = mod.get_function("make_noise")
    assign_biome = mod.get_function("assign_biome")
    make_stone_cuda = mod.get_function("make_stone")

    # Generate base terrain mesh
    heightmap = np.zeros(map_size_total).astype(np.float32)
    make_noise(
            drv.Out(heightmap),
            block=cuda_block_2d, grid=cuda_grid_shape
    )
    heightmap -= np.min(heightmap)
    heightmap /= np.max(heightmap)
    sea_level = np.median(heightmap)

    # Generate biomes
    biome_map = np.zeros((heightmap.shape[0], heightmap.shape[1], nbiomes)).astype(np.float32)
    assign_biome(
        drv.Out(biome_map), drv.In(heightmap), sea_level,
        block=cuda_block_2d, grid=cuda_grid_shape
    )
    #print(biome_map)
    biome_map /= np.sum(biome_map, axis=2, keepdims=True)

    def lerp(a, b, v):
        return a + (b-a) * v

    def point_on_surface(i, j):
        x = np.random.rand()
        z = np.random.rand()
        h = lerp(
                lerp(heightmap[i, j], heightmap[i+1, j], x),
                lerp(heightmap[i, j+1], heightmap[i+1, j+1], x),
                z
        )

        # RHS/LHS switch
        return np.array([
            -real_size[2] * 0.5 + (j+z) * (real_size[2] / map_size[1]),
            real_size[1] * (-sea_level + h),
            -real_size[0] * 0.5 + (i+x) * (real_size[0] / map_size[0])
        ])

    # Calculate where the vegetation will be places
    tree_density = np.zeros_like(heightmap, dtype=np.float32)
    get_tree_density = mod.get_function("get_tree_density")
    get_tree_density(
        drv.In(heightmap), sea_level, drv.In(biome_map), drv.Out(tree_density),
        block=cuda_block_2d, grid=cuda_grid_shape
    )
    vegetation_density = np.zeros_like(heightmap, dtype=np.float32)
    get_vegetation_density = mod.get_function("get_vegetation_density")
    get_vegetation_density(
        drv.In(heightmap), sea_level, drv.In(biome_map), drv.Out(vegetation_density),
        block=cuda_block_2d, grid=cuda_grid_shape
    )

    # Place some trees
    trees = []
    it = np.nditer(heightmap[:-1, :-1], flags=['multi_index'])
    while not it.finished:
        # Place trees
        density = tree_density[it.multi_index] / (map_scale*map_scale)
        if np.random.rand() < density:
            tree_type = 0
            tree_rnd = np.random.rand()
            if tree_rnd < 0.10:
                tree_type = 1
            elif tree_rnd < 0.12:
                tree_type = 2
            trees.append({
                'pos': point_on_surface( it.multi_index[0], it.multi_index[1] ),
                'rot': [ 0, np.random.rand() * 360, 0 ],
                'type': tree_type,
                'seed': np.random.randint(0, 1000000) #np.random.randint( 0, np.iinfo(np.int32).max )
            })
        # # Place grass
        # density = vegetation_density[it.multi_index] / (map_scale*map_scale)
        # while np.random.rand() < density:
        #     tree_type = 4
        #     trees.append({
        #         'pos': point_on_surface( it.multi_index[0], it.multi_index[1] ),
        #         'rot': [ 0, np.random.rand() * 360, 0 ],
        #         'type': 4,
        #         'seed': np.random.randint(0, 1000000) #np.random.randint( 0, np.iinfo(np.int32).max )
        #     })
        #     density -= 1.0
        it.iternext()

    meshes = []

    # Generate some stones
    nprop_prototypes = np.random.randint(20, 40)
    for i in np.arange(nprop_prototypes):
        verts, uvs, faces = make_stone(make_stone_cuda)
        meshes.append({
            'id': len(meshes),
            'verts': verts,
            'uvs': uvs,
            'faces': faces
        })

    stone_density = np.zeros_like(heightmap, dtype=np.float32)
    get_stone_density = mod.get_function("get_stone_density")
    get_stone_density(
        drv.In(heightmap), sea_level, drv.In(biome_map), drv.Out(stone_density),
        block=cuda_block_2d, grid=cuda_grid_shape
    )

    # Place the stones
    props = []
    it = np.nditer(heightmap[:-1, :-1], flags=['multi_index'])
    while not it.finished:
        # Place stones
        density = stone_density[it.multi_index] / (map_scale*map_scale)
        if np.random.rand() < density:
            mesh_id = np.random.randint(0, len(meshes))
            props.append({
                'mesh_id': mesh_id,
                'pos': point_on_surface( it.multi_index[0], it.multi_index[1] ) - 0.1,
                'rot': np.random.rand(3) * 360 #[ 0, 0, 0 ]
            })
        it.iternext()

    return heightmap, sea_level, biome_map, trees, meshes, props

computed_map = None
computed_map_hash = None
computed_mesh = None

def handle_get_map(map_request):
    """Process map generation request"""
    global computed_map, computed_map_hash
    map_scale = map_request.scale
    map_blocks_x = map_request.blocks_xz[0]
    map_blocks_z = map_request.blocks_xz[1]
    tex_size =  map_request.tex_size
    map_size = (map_blocks_x * tex_size, map_blocks_z * tex_size)
    map_hash = hash((map_scale, map_blocks_x, map_blocks_z, tex_size,
        map_request.block_size.x, map_request.block_size.y, map_request.block_size.z))

    real_size = (map_blocks_x * map_request.block_size.x, map_request.block_size.y, map_blocks_z * map_request.block_size.z)

    response = None

    if map_hash == computed_map_hash:
        print('Using precomputed map')
        response = computed_map
    else:
        print('Generating map...')
        np.random.seed(1)
        heightmap, sea_level, biome_map, trees, meshes, props = gen_map(map_size=map_size, map_scale=map_scale, real_size=real_size)

        # Pack the generated map into the protobuf object
        response = MapData()
        # Pack mesh geometries
        for mesh in meshes:
            mesh_data = response.meshes.add()
            mesh_data.id = mesh['id']
            for v in mesh['verts']:
                vert = mesh_data.verts.add()
                vert.x = v[0]
                vert.y = v[1]
                vert.z = v[2]
            for u in mesh['uvs']:
                uv = mesh_data.uvs.add()
                uv.x = u[0]
                uv.y = u[1]
            mesh_data.faces.extend( mesh['faces'].ravel() )

        for i in np.arange(map_blocks_x):
            for j in np.arange(map_blocks_z):
                map_block = response.blocks.add()
                # RHS/LHS switch
                map_block.idx.extend([ j, i ])
                # Pack terrain geometry
                theightmap = heightmap[
                    i*tex_size : (i+1)*tex_size+1,
                    j*tex_size : (j+1)*tex_size+1
                ]
                map_block.heightmap_shape.extend( theightmap.shape )
                map_block.heightmap.extend( theightmap.ravel() )
                # Pack biome data
                tbiomes = biome_map[
                    i*tex_size : (i+1)*tex_size,
                    j*tex_size : (j+1)*tex_size
                ]
                map_block.biomes_shape.extend( tbiomes.shape )
                map_block.biomes.extend( tbiomes.ravel() )
                # RHS/LHS switch
                map_block.pos.x = -real_size[2] * 0.5 + (j*map_request.block_size.z)
                map_block.pos.y = -sea_level * real_size[1]
                map_block.pos.z = -real_size[0] * 0.5 + (i*map_request.block_size.x)

                # RHS/LHS switch
                block_bounds = [
                    [ -real_size[2] * 0.5 + j * (real_size[2] / map_blocks_z), -real_size[2] * 0.5 + (j+1) * (real_size[2] / map_blocks_z) ],
                    [ -real_size[0] * 0.5 + i * (real_size[0] / map_blocks_x), -real_size[0] * 0.5 + (i+1) * (real_size[0] / map_blocks_x) ]
                ]
                # Pack the trees
                for tree in trees:
                    if tree['pos'][0] < block_bounds[0][0] or tree['pos'][0] >= block_bounds[0][1] \
                        or tree['pos'][2] < block_bounds[1][0] or tree['pos'][2] >= block_bounds[1][1]:
                        continue
                    tree_data = map_block.trees.add()
                    tree_data.pos.x = tree['pos'][0]
                    tree_data.pos.y = tree['pos'][1]
                    tree_data.pos.z = tree['pos'][2]
                    tree_data.rot.x = tree['rot'][0]
                    tree_data.rot.y = tree['rot'][1]
                    tree_data.rot.z = tree['rot'][2]
                    tree_data.type = tree['type']
                    tree_data.seed = tree['seed']

                # Pack props (stones)
                for prop in props:
                    if prop['pos'][0] < block_bounds[0][0] or prop['pos'][0] >= block_bounds[0][1] \
                        or prop['pos'][2] < block_bounds[1][0] or prop['pos'][2] >= block_bounds[1][1]:
                        continue
                    prop_data = map_block.props.add()
                    prop_data.mesh_id = prop['mesh_id']
                    prop_data.pos.x = prop['pos'][0]
                    prop_data.pos.y = prop['pos'][1]
                    prop_data.pos.z = prop['pos'][2]
                    prop_data.rot.x = prop['rot'][0]
                    prop_data.rot.y = prop['rot'][1]
                    prop_data.rot.z = prop['rot'][2]

        print('done')

        computed_map = response
        computed_map_hash = map_hash

    return response


class WorldGen(WorldGenServicer):
    def GetMap(self, request, context):
        return handle_get_map(request)

HOST, PORT = "", 7778
server = grpc.server( futures.ThreadPoolExecutor(max_workers=10), options=(
    ( cygrpc.ChannelArgKey.max_send_message_length, 1000000000 ),
    ( cygrpc.ChannelArgKey.max_receive_message_length, 1000000000 ),
) )
add_WorldGenServicer_to_server(WorldGen(), server)
server.add_insecure_port('[::]:%d' % PORT)
server.start()
print('Started gRPC on port %d' % PORT)

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
