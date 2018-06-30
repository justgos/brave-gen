# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: world_gen.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='world_gen.proto',
  package='worldgen',
  syntax='proto3',
  serialized_pb=_b('\n\x0fworld_gen.proto\x12\x08worldgen\" \n\x08Vector2_\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\"+\n\x08Vector3_\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"h\n\nMapRequest\x12\r\n\x05scale\x18\x01 \x01(\x02\x12\x11\n\tblocks_xz\x18\x02 \x03(\x05\x12\x10\n\x08tex_size\x18\x03 \x01(\x05\x12&\n\nblock_size\x18\x04 \x01(\x0b\x32\x12.worldgen.Vector3_\"h\n\x08TreeData\x12\x1f\n\x03pos\x18\x01 \x01(\x0b\x32\x12.worldgen.Vector3_\x12\x1f\n\x03rot\x18\x02 \x01(\x0b\x32\x12.worldgen.Vector3_\x12\x0c\n\x04type\x18\x03 \x01(\x05\x12\x0c\n\x04seed\x18\x04 \x01(\x05\"f\n\x05Mesh_\x12\n\n\x02id\x18\x01 \x01(\x05\x12!\n\x05verts\x18\x02 \x03(\x0b\x32\x12.worldgen.Vector3_\x12\x1f\n\x03uvs\x18\x03 \x03(\x0b\x32\x12.worldgen.Vector2_\x12\r\n\x05\x66\x61\x63\x65s\x18\x04 \x03(\x05\"Y\n\x04Prop\x12\x0f\n\x07mesh_id\x18\x01 \x01(\x05\x12\x1f\n\x03pos\x18\x02 \x01(\x0b\x32\x12.worldgen.Vector3_\x12\x1f\n\x03rot\x18\x03 \x01(\x0b\x32\x12.worldgen.Vector3_\"\xdf\x01\n\x08MapBlock\x12\x17\n\x0fheightmap_shape\x18\x01 \x03(\x05\x12\x11\n\theightmap\x18\x02 \x03(\x02\x12\x14\n\x0c\x62iomes_shape\x18\x03 \x03(\x05\x12\x0e\n\x06\x62iomes\x18\x04 \x03(\x02\x12\x11\n\tsea_level\x18\x05 \x01(\x02\x12\x1f\n\x03pos\x18\x06 \x01(\x0b\x32\x12.worldgen.Vector3_\x12\x0b\n\x03idx\x18\x07 \x03(\x05\x12!\n\x05trees\x18\x08 \x03(\x0b\x32\x12.worldgen.TreeData\x12\x1d\n\x05props\x18\t \x03(\x0b\x32\x0e.worldgen.Prop\"N\n\x07MapData\x12\x1f\n\x06meshes\x18\x01 \x03(\x0b\x32\x0f.worldgen.Mesh_\x12\"\n\x06\x62locks\x18\n \x03(\x0b\x32\x12.worldgen.MapBlock\"O\n\x0fSimulationFrame\x12\x10\n\x08humidity\x18\x01 \x03(\x02\x12\x15\n\rprecipitation\x18\x02 \x03(\x02\x12\x13\n\x0btemperature\x18\x03 \x03(\x02\"N\n\x0eSimulationData\x12\x11\n\tsim_shape\x18\x01 \x03(\x05\x12)\n\x06\x66rames\x18\n \x03(\x0b\x32\x19.worldgen.SimulationFrame2?\n\x08WorldGen\x12\x33\n\x06GetMap\x12\x14.worldgen.MapRequest\x1a\x11.worldgen.MapData\"\x00\x62\x06proto3')
)




_VECTOR2_ = _descriptor.Descriptor(
  name='Vector2_',
  full_name='worldgen.Vector2_',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='worldgen.Vector2_.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='worldgen.Vector2_.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=61,
)


_VECTOR3_ = _descriptor.Descriptor(
  name='Vector3_',
  full_name='worldgen.Vector3_',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='worldgen.Vector3_.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='worldgen.Vector3_.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='worldgen.Vector3_.z', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=63,
  serialized_end=106,
)


_MAPREQUEST = _descriptor.Descriptor(
  name='MapRequest',
  full_name='worldgen.MapRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='scale', full_name='worldgen.MapRequest.scale', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blocks_xz', full_name='worldgen.MapRequest.blocks_xz', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tex_size', full_name='worldgen.MapRequest.tex_size', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='block_size', full_name='worldgen.MapRequest.block_size', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=108,
  serialized_end=212,
)


_TREEDATA = _descriptor.Descriptor(
  name='TreeData',
  full_name='worldgen.TreeData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pos', full_name='worldgen.TreeData.pos', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rot', full_name='worldgen.TreeData.rot', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='worldgen.TreeData.type', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='seed', full_name='worldgen.TreeData.seed', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=214,
  serialized_end=318,
)


_MESH_ = _descriptor.Descriptor(
  name='Mesh_',
  full_name='worldgen.Mesh_',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='worldgen.Mesh_.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='verts', full_name='worldgen.Mesh_.verts', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='uvs', full_name='worldgen.Mesh_.uvs', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='faces', full_name='worldgen.Mesh_.faces', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=320,
  serialized_end=422,
)


_PROP = _descriptor.Descriptor(
  name='Prop',
  full_name='worldgen.Prop',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mesh_id', full_name='worldgen.Prop.mesh_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos', full_name='worldgen.Prop.pos', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rot', full_name='worldgen.Prop.rot', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=424,
  serialized_end=513,
)


_MAPBLOCK = _descriptor.Descriptor(
  name='MapBlock',
  full_name='worldgen.MapBlock',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='heightmap_shape', full_name='worldgen.MapBlock.heightmap_shape', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='heightmap', full_name='worldgen.MapBlock.heightmap', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='biomes_shape', full_name='worldgen.MapBlock.biomes_shape', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='biomes', full_name='worldgen.MapBlock.biomes', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sea_level', full_name='worldgen.MapBlock.sea_level', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos', full_name='worldgen.MapBlock.pos', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='idx', full_name='worldgen.MapBlock.idx', index=6,
      number=7, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='trees', full_name='worldgen.MapBlock.trees', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='props', full_name='worldgen.MapBlock.props', index=8,
      number=9, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=516,
  serialized_end=739,
)


_MAPDATA = _descriptor.Descriptor(
  name='MapData',
  full_name='worldgen.MapData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='meshes', full_name='worldgen.MapData.meshes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blocks', full_name='worldgen.MapData.blocks', index=1,
      number=10, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=741,
  serialized_end=819,
)


_SIMULATIONFRAME = _descriptor.Descriptor(
  name='SimulationFrame',
  full_name='worldgen.SimulationFrame',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='humidity', full_name='worldgen.SimulationFrame.humidity', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='precipitation', full_name='worldgen.SimulationFrame.precipitation', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='temperature', full_name='worldgen.SimulationFrame.temperature', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=821,
  serialized_end=900,
)


_SIMULATIONDATA = _descriptor.Descriptor(
  name='SimulationData',
  full_name='worldgen.SimulationData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sim_shape', full_name='worldgen.SimulationData.sim_shape', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frames', full_name='worldgen.SimulationData.frames', index=1,
      number=10, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=902,
  serialized_end=980,
)

_MAPREQUEST.fields_by_name['block_size'].message_type = _VECTOR3_
_TREEDATA.fields_by_name['pos'].message_type = _VECTOR3_
_TREEDATA.fields_by_name['rot'].message_type = _VECTOR3_
_MESH_.fields_by_name['verts'].message_type = _VECTOR3_
_MESH_.fields_by_name['uvs'].message_type = _VECTOR2_
_PROP.fields_by_name['pos'].message_type = _VECTOR3_
_PROP.fields_by_name['rot'].message_type = _VECTOR3_
_MAPBLOCK.fields_by_name['pos'].message_type = _VECTOR3_
_MAPBLOCK.fields_by_name['trees'].message_type = _TREEDATA
_MAPBLOCK.fields_by_name['props'].message_type = _PROP
_MAPDATA.fields_by_name['meshes'].message_type = _MESH_
_MAPDATA.fields_by_name['blocks'].message_type = _MAPBLOCK
_SIMULATIONDATA.fields_by_name['frames'].message_type = _SIMULATIONFRAME
DESCRIPTOR.message_types_by_name['Vector2_'] = _VECTOR2_
DESCRIPTOR.message_types_by_name['Vector3_'] = _VECTOR3_
DESCRIPTOR.message_types_by_name['MapRequest'] = _MAPREQUEST
DESCRIPTOR.message_types_by_name['TreeData'] = _TREEDATA
DESCRIPTOR.message_types_by_name['Mesh_'] = _MESH_
DESCRIPTOR.message_types_by_name['Prop'] = _PROP
DESCRIPTOR.message_types_by_name['MapBlock'] = _MAPBLOCK
DESCRIPTOR.message_types_by_name['MapData'] = _MAPDATA
DESCRIPTOR.message_types_by_name['SimulationFrame'] = _SIMULATIONFRAME
DESCRIPTOR.message_types_by_name['SimulationData'] = _SIMULATIONDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Vector2_ = _reflection.GeneratedProtocolMessageType('Vector2_', (_message.Message,), dict(
  DESCRIPTOR = _VECTOR2_,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.Vector2_)
  ))
_sym_db.RegisterMessage(Vector2_)

Vector3_ = _reflection.GeneratedProtocolMessageType('Vector3_', (_message.Message,), dict(
  DESCRIPTOR = _VECTOR3_,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.Vector3_)
  ))
_sym_db.RegisterMessage(Vector3_)

MapRequest = _reflection.GeneratedProtocolMessageType('MapRequest', (_message.Message,), dict(
  DESCRIPTOR = _MAPREQUEST,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.MapRequest)
  ))
_sym_db.RegisterMessage(MapRequest)

TreeData = _reflection.GeneratedProtocolMessageType('TreeData', (_message.Message,), dict(
  DESCRIPTOR = _TREEDATA,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.TreeData)
  ))
_sym_db.RegisterMessage(TreeData)

Mesh_ = _reflection.GeneratedProtocolMessageType('Mesh_', (_message.Message,), dict(
  DESCRIPTOR = _MESH_,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.Mesh_)
  ))
_sym_db.RegisterMessage(Mesh_)

Prop = _reflection.GeneratedProtocolMessageType('Prop', (_message.Message,), dict(
  DESCRIPTOR = _PROP,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.Prop)
  ))
_sym_db.RegisterMessage(Prop)

MapBlock = _reflection.GeneratedProtocolMessageType('MapBlock', (_message.Message,), dict(
  DESCRIPTOR = _MAPBLOCK,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.MapBlock)
  ))
_sym_db.RegisterMessage(MapBlock)

MapData = _reflection.GeneratedProtocolMessageType('MapData', (_message.Message,), dict(
  DESCRIPTOR = _MAPDATA,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.MapData)
  ))
_sym_db.RegisterMessage(MapData)

SimulationFrame = _reflection.GeneratedProtocolMessageType('SimulationFrame', (_message.Message,), dict(
  DESCRIPTOR = _SIMULATIONFRAME,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.SimulationFrame)
  ))
_sym_db.RegisterMessage(SimulationFrame)

SimulationData = _reflection.GeneratedProtocolMessageType('SimulationData', (_message.Message,), dict(
  DESCRIPTOR = _SIMULATIONDATA,
  __module__ = 'world_gen_pb2'
  # @@protoc_insertion_point(class_scope:worldgen.SimulationData)
  ))
_sym_db.RegisterMessage(SimulationData)



_WORLDGEN = _descriptor.ServiceDescriptor(
  name='WorldGen',
  full_name='worldgen.WorldGen',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=982,
  serialized_end=1045,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetMap',
    full_name='worldgen.WorldGen.GetMap',
    index=0,
    containing_service=None,
    input_type=_MAPREQUEST,
    output_type=_MAPDATA,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_WORLDGEN)

DESCRIPTOR.services_by_name['WorldGen'] = _WORLDGEN

# @@protoc_insertion_point(module_scope)
