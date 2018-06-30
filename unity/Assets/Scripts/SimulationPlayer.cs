using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

using Worldgen;

[ExecuteInEditMode]
public class SimulationPlayer : MonoBehaviour
{
    public Material temperatureMat;
    public Material humidityMat;
    public float fps = 10.0f;

    ComputeBuffer quadPoints;
    ComputeBuffer quadUVs;
    int[] simShape;
    List<ComputeBuffer> temperatureBuffers;
    List<ComputeBuffer> humidityBuffers;
    int curFrame;
    float lastFrameTime;

    public struct BGParticle
    {
        public Vector3 position;
        public float val;
    }

    void Start()
    {
        //
    }

    void Update()
    {
        if (temperatureBuffers == null)
        {
            quadPoints = new ComputeBuffer(6, Marshal.SizeOf(new Vector3()));
                quadPoints.SetData(new[] {
                new Vector3(-0.5f, 0.5f),
                new Vector3(0.5f, 0.5f),
                new Vector3(0.5f, -0.5f),
                new Vector3(0.5f, -0.5f),
                new Vector3(-0.5f, -0.5f),
                new Vector3(-0.5f, 0.5f),
            });

            quadUVs = new ComputeBuffer(6, Marshal.SizeOf(new Vector2()));
                quadUVs.SetData(new[] {
                new Vector2(0.0f, 1.0f),
                new Vector2(1.0f, 1.0f),
                new Vector2(1.0f, 0.0f),
                new Vector2(1.0f, 0.0f),
                new Vector2(0.0f, 0.0f),
                new Vector2(0.0f, 1.0f),
            });

            // Load simulation frames
            temperatureBuffers = new List<ComputeBuffer>();
            humidityBuffers = new List<ComputeBuffer>();
            var simData = SimulationData.Parser.ParseFrom(File.ReadAllBytes("..\\python\\sim_data.dat"));
            simShape = new int[] { simData.SimShape[0], simData.SimShape[1], simData.SimShape[2] };

            // Looup terrain instance to get its height and vertical position
            var terrainBlock = GameObject.FindObjectsOfType<Terrain>()[0];
            var terrainHeight = terrainBlock.terrainData.size.y;
            var p = new BGParticle();
            Debug.Log("Simulation shape: " + simShape[0] + ", " + simShape[1] + ", " + simShape[2]);
            for (var f = 0; f < simData.Frames.Count; f++)
            {
                var tPoints = new List<BGParticle>();
                var hPoints = new List<BGParticle>();
                for (var i = 0; i < simShape[0]; i++)
                {
                    for (var j = 0; j < simShape[1]; j++)
                    {
                        for (var k = 0; k < simShape[2]; k++)
                        {
                            var t = simData.Frames[f].Temperature[i * simShape[1] * simShape[2] + j * simShape[2] + k];
                            tPoints.Add(new BGParticle
                            {
                                position = new Vector3(
                                    (((float)j)/simShape[0]-0.5f) * 1000,
                                    (((float)k)/simShape[2]*1.3f - 0.1f) * terrainHeight + terrainBlock.transform.position.y,
                                    (((float)i)/simShape[1]-0.5f) * 1000
                                ),
                                val = t,
                            });

                            var h = simData.Frames[f].Humidity[i * simShape[1] * simShape[2] + j * simShape[2] + k];
                            hPoints.Add(new BGParticle
                            {
                                position = new Vector3(
                                    (((float)j) / simShape[0] - 0.5f) * 1000,
                                    (((float)k) / simShape[2] * 1.3f - 0.1f) * terrainHeight + terrainBlock.transform.position.y,
                                    (((float)i) / simShape[1] - 0.5f) * 1000
                                ),
                                val = h,
                            });
                        }
                    }
                }
                var temperatureBuffer = new ComputeBuffer(tPoints.Count, Marshal.SizeOf(p), ComputeBufferType.Default);
                temperatureBuffer.SetData(tPoints.ToArray());
                temperatureBuffers.Add(temperatureBuffer);
                var humidityBuffer = new ComputeBuffer(hPoints.Count, Marshal.SizeOf(p), ComputeBufferType.Default);
                humidityBuffer.SetData(hPoints.ToArray());
                humidityBuffers.Add(humidityBuffer);
            }

            curFrame = 0;
            lastFrameTime = Time.time;
        }
    }

    void OnRenderObject()
    {
        if (temperatureBuffers == null)
            return;

        if (Time.time > lastFrameTime + 1.0f / fps)
        {
            curFrame = (curFrame + 1) % temperatureBuffers.Count;
            lastFrameTime = Time.time;
        }

        // Render temperatures
        temperatureMat.SetBuffer("quadPoints", quadPoints);
        temperatureMat.SetBuffer("quadUVs", quadUVs);
        temperatureMat.SetMatrix("baseTransform", Matrix4x4.identity);
        temperatureMat.SetFloat("cutoffValue", 0.1f);

        var temperatureBuffer = temperatureBuffers[curFrame];
        temperatureMat.SetBuffer("particles", temperatureBuffer);
        temperatureMat.SetPass(0);
        Graphics.DrawProcedural(MeshTopology.Triangles, 6, temperatureBuffer.count);

        // Render humidity
        humidityMat.SetBuffer("quadPoints", quadPoints);
        humidityMat.SetBuffer("quadUVs", quadUVs);
        humidityMat.SetMatrix("baseTransform", Matrix4x4.identity);
        humidityMat.SetFloat("cutoffValue", 0.01f);

        var humidityBuffer = humidityBuffers[curFrame];
        humidityMat.SetBuffer("particles", humidityBuffer);
        humidityMat.SetPass(0);
        Graphics.DrawProcedural(MeshTopology.Triangles, 6, humidityBuffer.count);
    }

    void Destroy()
    {
        if (temperatureBuffers != null)
        {
            foreach (var temperatureBuffer in temperatureBuffers)
            {
                temperatureBuffer.Release();
            }
            foreach (var humidityBuffer in humidityBuffers)
            {
                humidityBuffer.Release();
            }
        }
    }
}
