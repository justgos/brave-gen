@echo off
pushd .
ECHO Generating Python protos..
python -m grpc_tools.protoc -I%~dp0\proto\ --python_out=%~dp0\python\ %~dp0\proto\world_gen.proto

ECHO Generating Unity protos..
%~dp0\tools\grpc-protoc\win\protoc.exe -I%~dp0\proto\ --csharp_out %~dp0\unity\Assets\Scripts\proto\ --grpc_out %~dp0\unity\Assets\Scripts\proto\ --plugin=protoc-gen-grpc=%~dp0\tools\grpc-protoc\win\grpc_csharp_plugin.exe %~dp0\proto\world_gen.proto

popd
ECHO Done
