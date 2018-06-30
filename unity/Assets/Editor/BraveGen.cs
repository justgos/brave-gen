using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

using System;

using Grpc.Core;

using Worldgen;

public class BraveGen : MonoBehaviour {
    //private const string GEN_HOST = "192.168.0.101";
    private const string GEN_HOST = "localhost";

    private static float mapScale = 0.4f;
    private static int mapBlocksX = 2;
    private static int mapBlocksZ = 2;
    private static int texSize = 1024;
    private static int alphasSize = 1024;
    private static float blockScale = 500;
    private static float blockHeight = 170;

    private static object externalLock = new object();
    private static Action externalCallbacks;

    private static object mapReadyLock = new object();
    private static Action<byte[]> mapReadyCallbacks;

    private static Terrain[,] terrainBlocks = new Terrain[1,1];
    private static List<GameObject> terrainObjs = new List<GameObject>();
    private static Dictionary<int, Mesh> meshes = new Dictionary<int, Mesh>();
    private static List<GameObject> props = new List<GameObject>();

    void Start () {
        //
    }

    [MenuItem("BraveGen/Generate Terrain")]
    static public void GenerateTerrain() {
        Debug.Log("Generating map...");
        
        var channel = new Channel(GEN_HOST + ":7778", ChannelCredentials.Insecure, 
            new List<ChannelOption>() {
                new ChannelOption(ChannelOptions.MaxSendMessageLength, 1000000000),
                new ChannelOption(ChannelOptions.MaxReceiveMessageLength, 1000000000)
        });

        var compressionMetadata = new Metadata
        {
            { new Metadata.Entry("grpc-internal-encoding-request", "gzip") }
        };
        var client = new WorldGen.WorldGenClient(channel);

        var mapRequest = new MapRequest();
        mapRequest.Scale = mapScale;
        mapRequest.BlocksXz.AddRange(new int[] { mapBlocksX, mapBlocksZ });
        mapRequest.TexSize = texSize;
        mapRequest.BlockSize = new Vector3_();
        mapRequest.BlockSize.X = blockScale;
        mapRequest.BlockSize.Y = blockHeight;
        mapRequest.BlockSize.Z = blockScale;

        try {
            var map = client.GetMap(mapRequest, new CallOptions(compressionMetadata));
            handleGetMap(map);
        } catch(RpcException e) {
            Debug.LogError(e);
        }

        channel.ShutdownAsync().Wait();
    }

    [MenuItem("BraveGen/ShowTreeInfo")]
    static public void ShowTreeInfo() {
        var tree = (Selection.activeObject as GameObject).GetComponent<Terrain>().terrainData.treeInstances[0];
        Debug.Log(tree.position);
        Debug.Log(tree.prototypeIndex);
    }
    
    private static T[,] Make2DArray<T>(Google.Protobuf.Collections.RepeatedField<T> input, int height, int width) {
        T[,] output = new T[width, height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                output[i, j] = input[i * height + j];
            }
        }
        return output;
    }

    private static T[,,] Make3DArray<T>(Google.Protobuf.Collections.RepeatedField<T> input, int height, int width, int depth) {
        T[,,] output = new T[width, height, depth];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < depth; k++) {
                    output[i, j, k] = input[i * (height*depth) + j * depth + k];
                }
            }
        }
        return output;
    }
    
    static private void handleGetMap(MapData map) {
        StaticAssetStorage.instance.ExternalInvoke(() => {
            if (props != null) {
                foreach (var prop in props) {
                    if (prop)
                        DestroyImmediate(prop);
                }
                props = null;
            }
            /*foreach (var mesh in meshes) {
                DestroyImmediate(mesh.Value);
            }*/
            meshes = null;
            if (terrainBlocks != null) {
                foreach (var block in terrainBlocks) {
                    if (block)
                        DestroyImmediate(block);
                }
                terrainBlocks = null;
            }

            meshes = new Dictionary<int, Mesh>();
            foreach (var meshData in map.Meshes) {
                var mesh = new Mesh();

                var verts = new Vector3[meshData.Verts.Count];
                for (var i = 0; i < meshData.Verts.Count; i++)
                    verts[i] = new Vector3(meshData.Verts[i].X, meshData.Verts[i].Y, meshData.Verts[i].Z);
                var faces = new int[meshData.Faces.Count];
                for (var i = 0; i < meshData.Faces.Count; i++)
                    faces[i] = meshData.Faces[i];

                mesh.vertices = verts;
                mesh.SetIndices(faces, MeshTopology.Triangles, 0);
                mesh.RecalculateNormals();
                meshes.Add(meshData.Id, mesh);
            }

            props = new List<GameObject>();

            terrainBlocks = new Terrain[mapBlocksX, mapBlocksZ];
            for (var b = 0; b < map.Blocks.Count; b++) {
                var mapBlock = map.Blocks[b];
                
                BuildMap(
                    mapBlock.Idx[0], mapBlock.Idx[1],
                    Make2DArray(mapBlock.Heightmap, mapBlock.HeightmapShape[0], mapBlock.HeightmapShape[1]),
                    Make3DArray(mapBlock.Biomes, alphasSize, alphasSize, mapBlock.BiomesShape[2]),
                    new Vector3(mapBlock.Pos.X, mapBlock.Pos.Y, mapBlock.Pos.Z),
                    mapBlock.Trees, mapBlock.Props
                );
            }
        });
        // Editor button mode - end connection
        //socket.Close();
    }

    static public void ExternalInvoke(System.Action callback) {
        lock (externalLock) {
            externalCallbacks += callback;
        }
    }

    static public void SubscribeMapReady(System.Action<byte[]> callback) {
        lock (mapReadyLock) {
            mapReadyCallbacks += callback;
        }
    }

    static void BuildMap(int blocki, int blockj, float[,] heightmap, float[,,] biomes, Vector3 pos, 
                            IEnumerable<TreeData> trees, IEnumerable<Prop> propInstances) {
        var terrain = Instantiate(StaticAssetStorage.instance.baseTerrain);
        //var terrainData = Instantiate(terrain.terrainData);
        var terrainData = TerrainDataCloner.Clone(terrain.terrainData);
        AssetDatabase.CreateAsset(terrainData, String.Format("Assets/Terrain Data/block-{0}-{1}.asset", blocki, blockj));
        terrainData.heightmapResolution = texSize + 1;
        terrainData.size = new Vector3(blockScale, blockHeight, blockScale);
        terrainData.SetHeights(0, 0, heightmap);
        terrainData.alphamapResolution = alphasSize;
        terrainData.SetAlphamaps(0, 0, biomes);
        terrain.terrainData = terrainData;
        terrain.GetComponent<TerrainCollider>().terrainData = terrainData;
        terrain.transform.position = pos;
        terrainBlocks[blocki, blockj] = terrain;

        for (var i = 0; i < mapBlocksX; i++) {
            for (var j = 0; j < mapBlocksZ; j++) {
                if (terrainBlocks[i, j] == null)
                    continue;
                terrainBlocks[i, j].SetNeighbors(
                    i > 0 ? terrainBlocks[i - 1, j] : null,
                    j < mapBlocksZ - 1 ? terrainBlocks[i, j + 1] : null,
                    i < mapBlocksX - 1 ? terrainBlocks[i + 1, j] : null,
                    j > 0 ? terrainBlocks[i, j - 1] : null
                );
            }
        }

        List<TreeInstance> treeInstances = new List<TreeInstance>();
        foreach( var treeData in trees ) {
            var tree = Instantiate(StaticAssetStorage.instance.trees[treeData.Type]);
            tree.transform.position = new Vector3(treeData.Pos.X, treeData.Pos.Y, treeData.Pos.Z);
            tree.transform.rotation = Quaternion.Euler(treeData.Rot.X, treeData.Rot.Y, treeData.Rot.Z);
            if(tree.transform.GetComponentInChildren<Tree>() != null
                && (tree.transform.GetComponentInChildren<Tree>().data as TreeEditor.TreeData) != null)
                (tree.transform.GetComponentInChildren<Tree>().data as TreeEditor.TreeData).root.seed = treeData.Seed;
            //(tree.GetComponent<Tree>().data as TreeEditor.TreeData).root.UpdateSeed();
            tree.transform.parent = terrain.transform;
        }
        terrainData.treeInstances = treeInstances.ToArray();
        
        foreach(var propInstance in propInstances) {
            var prop = Instantiate(StaticAssetStorage.instance.baseProp);

            prop.GetComponent<MeshFilter>().sharedMesh = meshes[propInstance.MeshId];
            prop.GetComponent<MeshRenderer>().sharedMaterials.SetValue(StaticAssetStorage.instance.terrainMaterial, 0);
            prop.GetComponent<MeshRenderer>().sharedMaterial = prop.GetComponent<MeshRenderer>().sharedMaterials[0];
            prop.GetComponent<MeshCollider>().sharedMesh = prop.GetComponent<MeshFilter>().sharedMesh;

            prop.transform.position = new Vector3(propInstance.Pos.X, propInstance.Pos.Y, propInstance.Pos.Z);
            prop.transform.rotation = Quaternion.Euler(propInstance.Rot.X, propInstance.Rot.Y, propInstance.Rot.Z);

            prop.transform.parent = terrain.transform;
            props.Add(prop);
        }

        terrain.Flush();
    }
}
