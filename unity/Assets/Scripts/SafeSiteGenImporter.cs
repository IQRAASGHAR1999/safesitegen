// SafeSiteGenImporter.cs
//
// Builds a Unity scene from a SafeSiteGen scene contract (scene_unity.json).
//
// The contract is deliberately flat: one record per instance with a transform,
// a prefab key and the hazard metadata the runtime needs to score a response.
// Unity never has to understand the Hazard Scenario Graph, and it never
// re-decides whether something is a hazard: that was settled by the validation
// gate before this file was written.
//
// Setup
// -----
//   1. New Unity project, 3D (Built-in or URP both work).
//   2. Put this file in Assets/Scripts/.
//   3. Put scene_unity.json in Assets/StreamingAssets/.
//   4. Create an empty GameObject, add SafeSiteGenImporter, press Play.
//
// With no prefabs assigned it builds the scene from primitives, which is enough
// to demonstrate the pipeline. Assign real prefabs in the PrefabBinding list to
// swap them in without touching the code.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace SafeSiteGen
{
    // ----------------------------------------------------------------- DTOs
    [Serializable] public class Vec3Holder { public float[] position; public float rotationY; }

    [Serializable]
    public class Scoring
    {
        public string hazardId;
        public string hazardClass;
        public string clause;
        public string severity;
        public string[] requiredControls;
        public float cueSalience;
    }

    [Serializable]
    public class Instance
    {
        public string instanceId;
        public string prefab;
        public string assetType;
        public string kind;
        public Vec3Holder transform;
        public string zone;
        public bool isDistractor;
        public string[] controls;
        public Scoring scoring;          // null for everything that is not a teaching point
    }

    [Serializable] public class Zone { public string name; public string type; public int x, y, w, d; }
    [Serializable] public class Obstacle { public int x, y, w, d; public string asset_type; }

    [Serializable]
    public class Site
    {
        public string id, name, provenance;
        public int width, depth;
        public float grade_elevation_ft, deck_elevation_ft;
        public Zone[] zones;
        public Obstacle[] obstacles;
        public int[] entry;
        public PatrolPoint[] patrol;
    }

    [Serializable] public class PatrolPoint { public int x, y; }

    [Serializable] public class AnswerEntry { public string hazardId, entityId, hazardClass, clause; }

    [Serializable]
    public class SceneContract
    {
        public string formatVersion;
        public string scenarioId;
        public Site site;
        public string trade, activity;
        public float difficulty;
        public bool validated;
        public Instance[] instances;
        public AnswerEntry[] answerKey;
    }

    [Serializable]
    public class PrefabBinding
    {
        public string assetType;
        public GameObject prefab;
    }

    // ------------------------------------------------------------- importer
    public class SafeSiteGenImporter : MonoBehaviour
    {
        [Header("Input")]
        [Tooltip("File name inside Assets/StreamingAssets/")]
        public string sceneFile = "scene_unity.json";

        [Header("Optional prefab bindings (falls back to primitives)")]
        public List<PrefabBinding> prefabBindings = new List<PrefabBinding>();

        [Header("Build options")]
        public bool buildGround = true;
        public bool drawTraineeRoute = true;
        public bool spawnFirstPersonRig = true;

        public SceneContract Contract { get; private set; }

        private readonly Dictionary<string, GameObject> _spawned = new Dictionary<string, GameObject>();

        void Start() { Build(); }

        public void Build()
        {
            string path = Path.Combine(Application.streamingAssetsPath, sceneFile);
            if (!File.Exists(path))
            {
                Debug.LogError($"[SafeSiteGen] scene contract not found at {path}");
                return;
            }

            Contract = JsonUtility.FromJson<SceneContract>(File.ReadAllText(path));

            if (!Contract.validated)
            {
                // A scene that never cleared the gate must not be presented as training.
                Debug.LogError($"[SafeSiteGen] {Contract.scenarioId} was rejected by the validation " +
                               "gate. Refusing to build it as a training environment.");
                return;
            }

            Debug.Log($"[SafeSiteGen] {Contract.scenarioId} on {Contract.site.name}, " +
                      $"{Contract.trade} / {Contract.activity}, difficulty {Contract.difficulty:F2}, " +
                      $"{Contract.answerKey.Length} teaching points");

            if (buildGround) BuildSite();
            foreach (var inst in Contract.instances) SpawnInstance(inst);
            if (drawTraineeRoute) DrawRoute();
            if (spawnFirstPersonRig) SpawnRig();
        }

        // ------------------------------------------------------------- site
        void BuildSite()
        {
            var root = new GameObject("Site").transform;
            root.SetParent(transform, false);

            foreach (var z in Contract.site.zones)
            {
                var quad = GameObject.CreatePrimitive(PrimitiveType.Cube);
                quad.name = $"Zone_{z.name}_{z.type}";
                quad.transform.SetParent(root, false);
                quad.transform.localPosition = new Vector3(z.x + z.w / 2f, -0.05f, z.y + z.d / 2f);
                quad.transform.localScale = new Vector3(z.w, 0.1f, z.d);
                quad.GetComponent<Renderer>().material.color = ZoneColour(z.type);
            }

            foreach (var o in Contract.site.obstacles)
            {
                var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
                box.name = $"Fixed_{o.asset_type}";
                box.transform.SetParent(root, false);
                box.transform.localPosition = new Vector3(o.x + o.w / 2f, 0.95f, o.y + o.d / 2f);
                box.transform.localScale = new Vector3(o.w, 1.9f, o.d);
                box.GetComponent<Renderer>().material.color = new Color(0.30f, 0.34f, 0.38f);
            }
        }

        static Color ZoneColour(string type)
        {
            switch (type)
            {
                case "slab_edge":    return new Color(0.36f, 0.31f, 0.24f);
                case "excavation":   return new Color(0.29f, 0.25f, 0.21f);
                case "scaffold_bay": return new Color(0.33f, 0.27f, 0.33f);
                case "haul_road":    return new Color(0.20f, 0.22f, 0.25f);
                case "laydown":      return new Color(0.33f, 0.31f, 0.26f);
                case "work_zone":    return new Color(0.22f, 0.26f, 0.32f);
                case "access":       return new Color(0.22f, 0.29f, 0.24f);
                default:             return new Color(0.24f, 0.28f, 0.32f);
            }
        }

        // -------------------------------------------------------- instances
        void SpawnInstance(Instance inst)
        {
            GameObject go = null;
            var binding = prefabBindings.Find(b => b.assetType == inst.assetType && b.prefab != null);

            if (binding != null)
            {
                go = Instantiate(binding.prefab, transform);
            }
            else
            {
                go = GameObject.CreatePrimitive(
                    inst.kind == "worker" ? PrimitiveType.Capsule : PrimitiveType.Cube);
                go.transform.SetParent(transform, false);
                var s = PrimitiveScale(inst.kind);
                go.transform.localScale = s;
                go.GetComponent<Renderer>().material.color = KindColour(inst.kind);
            }

            float y = inst.kind == "worker" ? 0.9f : PrimitiveScale(inst.kind).y / 2f;
            go.name = $"{inst.instanceId}_{inst.assetType}";
            go.transform.localPosition = new Vector3(
                inst.transform.position[0] + 0.5f, y, inst.transform.position[2] + 0.5f);
            go.transform.localRotation = Quaternion.Euler(0f, inst.transform.rotationY, 0f);

            // Every instance gets a marker; only the ones with scoring are hazards.
            var marker = go.AddComponent<HazardMarker>();
            marker.instanceId = inst.instanceId;
            marker.assetType = inst.assetType;
            marker.isDistractor = inst.isDistractor;
            marker.scoring = inst.scoring;

            if (go.GetComponent<Collider>() == null) go.AddComponent<BoxCollider>();
            _spawned[inst.instanceId] = go;
        }

        static Vector3 PrimitiveScale(string kind)
        {
            switch (kind)
            {
                case "worker":      return new Vector3(0.55f, 0.9f, 0.55f);
                case "equipment":   return new Vector3(1.3f, 2.2f, 1.3f);
                case "control":     return new Vector3(1.6f, 1.05f, 0.3f);
                case "environment": return new Vector3(0.4f, 4.5f, 0.4f);
                default:            return new Vector3(1.1f, 1.1f, 1.1f);
            }
        }

        static Color KindColour(string kind)
        {
            switch (kind)
            {
                case "worker":      return new Color(0.91f, 0.72f, 0.29f);
                case "equipment":   return new Color(0.50f, 0.65f, 0.79f);
                case "control":     return new Color(0.44f, 0.75f, 0.56f);
                case "environment": return new Color(0.72f, 0.61f, 0.81f);
                default:            return new Color(0.64f, 0.68f, 0.72f);
            }
        }

        // ------------------------------------------------------------ route
        void DrawRoute()
        {
            if (Contract.site.patrol == null || Contract.site.patrol.Length < 2) return;
            var go = new GameObject("TraineeRoute");
            go.transform.SetParent(transform, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.positionCount = Contract.site.patrol.Length;
            lr.widthMultiplier = 0.12f;
            lr.material = new Material(Shader.Find("Sprites/Default"));
            lr.startColor = lr.endColor = new Color(0.55f, 0.44f, 0.24f);
            for (int i = 0; i < Contract.site.patrol.Length; i++)
            {
                var p = Contract.site.patrol[i];
                lr.SetPosition(i, new Vector3(p.x + 0.5f, 0.08f, p.y + 0.5f));
            }
        }

        void SpawnRig()
        {
            var cam = Camera.main;
            if (cam == null)
            {
                var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
                cam = camGo.AddComponent<Camera>();
            }
            var e = Contract.site.entry;
            cam.transform.position = new Vector3(e[0] + 0.5f, 1.68f, e[1] + 0.5f);
            cam.transform.rotation = Quaternion.identity;
            if (cam.GetComponent<TraineeController>() == null)
                cam.gameObject.AddComponent<TraineeController>().importer = this;
        }

        public IEnumerable<AnswerEntry> AnswerKey => Contract != null
            ? (IEnumerable<AnswerEntry>)Contract.answerKey : new AnswerEntry[0];
    }

    // --------------------------------------------------------------- marker
    public class HazardMarker : MonoBehaviour
    {
        public string instanceId;
        public string assetType;
        public bool isDistractor;
        public Scoring scoring;

        // JsonUtility materialises a null JSON object as an empty instance rather
        // than null, so emptiness is the real test, not reference equality.
        public bool IsHazard => scoring != null && !string.IsNullOrEmpty(scoring.hazardId);
    }
}
