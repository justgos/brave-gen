using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[ExecuteInEditMode]
public class StaticAssetStorage : MonoBehaviour {
    public static StaticAssetStorage instance;

    private object externalLock = new object();
    private Action externalCallbacks;

    public GameObject terrainPrefab;
    public Material terrainMaterial;
    public Terrain baseTerrain;
    public GameObject[] trees;
    public GameObject baseProp;

    void Awake() {
        instance = this;
    }

    void OnEnable() {
        EditorApplication.update += UpdateInEditor;
    }

    void OnDisable() {
        EditorApplication.update -= UpdateInEditor;
    }

    public void ExternalInvoke(System.Action callback) {
        lock (externalLock) {
            externalCallbacks += callback;
        }
    }

    void UpdateInEditor() {
        if (instance == null)
            instance = this;

        Action a = null;
        lock (externalLock) {
            if (externalCallbacks != null) {
                a = externalCallbacks;
                externalCallbacks = null;
            }
        }
        if (a != null) a();
    }
}
