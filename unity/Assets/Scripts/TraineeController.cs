// TraineeController.cs
//
// Minimal first-person trainee loop: walk the site, click what you believe is a
// hazard, get scored against the answer key the validation gate verified.
//
// Attach to the main camera (SafeSiteGenImporter does this automatically) and
// press Play. WASD to move, hold right mouse to look, left click to flag,
// Enter to finish.

using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace SafeSiteGen
{
    public class TraineeController : MonoBehaviour
    {
        public SafeSiteGenImporter importer;
        public float walkSpeed = 3.2f;
        public float runSpeed = 6.5f;
        public float lookSpeed = 2.4f;
        public float maxFlagDistance = 40f;

        private readonly HashSet<string> _flagged = new HashSet<string>();
        private readonly List<string> _log = new List<string>();
        private float _yaw, _pitch;
        private bool _finished;

        void Start()
        {
            var e = transform.eulerAngles;
            _yaw = e.y;
            _pitch = e.x;
        }

        void Update()
        {
            if (_finished)
            {
                if (Input.GetKeyDown(KeyCode.R)) Restart();
                return;
            }

            Look();
            Move();

            if (Input.GetMouseButtonDown(0)) Flag();
            if (Input.GetKeyDown(KeyCode.Return)) Finish();
        }

        void Look()
        {
            if (!Input.GetMouseButton(1)) return;
            _yaw += Input.GetAxis("Mouse X") * lookSpeed;
            _pitch = Mathf.Clamp(_pitch - Input.GetAxis("Mouse Y") * lookSpeed, -80f, 80f);
            transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
        }

        void Move()
        {
            float speed = Input.GetKey(KeyCode.LeftShift) ? runSpeed : walkSpeed;
            var dir = new Vector3(Input.GetAxisRaw("Horizontal"), 0f, Input.GetAxisRaw("Vertical"));
            if (dir.sqrMagnitude < 0.01f) return;
            var world = transform.TransformDirection(dir.normalized);
            world.y = 0f;
            transform.position += world.normalized * speed * Time.deltaTime;
            var p = transform.position;
            transform.position = new Vector3(p.x, 1.68f, p.z);
        }

        void Flag()
        {
            var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (!Physics.Raycast(ray, out var hit, maxFlagDistance)) return;

            var marker = hit.collider.GetComponentInParent<HazardMarker>();
            if (marker == null || _flagged.Contains(marker.instanceId)) return;

            _flagged.Add(marker.instanceId);

            if (marker.IsHazard)
            {
                _log.Add($"FOUND   {marker.scoring.hazardClass}  ({marker.scoring.clause})");
                Debug.Log($"[trainee] correct: {marker.scoring.hazardClass} " +
                          $"violates {marker.scoring.clause}. Required control: " +
                          $"{string.Join(", ", marker.scoring.requiredControls)}");
                Tint(marker.gameObject, new Color(0.22f, 0.70f, 0.43f));
            }
            else
            {
                _log.Add($"FALSE   {marker.assetType} is compliant");
                Debug.Log($"[trainee] false alarm: {marker.assetType} is compliant here");
                Tint(marker.gameObject, new Color(0.91f, 0.64f, 0.24f));
            }
        }

        static void Tint(GameObject go, Color c)
        {
            var r = go.GetComponentInChildren<Renderer>();
            if (r != null) r.material.color = c;
        }

        void Finish()
        {
            _finished = true;
            var key = importer != null ? importer.AnswerKey.ToList() : new List<AnswerEntry>();
            var missed = key.Where(a => !_flagged.Contains(a.entityId)).ToList();
            int found = key.Count - missed.Count;

            var sb = new StringBuilder();
            sb.AppendLine($"=== {found}/{key.Count} hazards identified ===");
            foreach (var line in _log) sb.AppendLine(line);
            foreach (var a in missed) sb.AppendLine($"MISSED  {a.hazardClass}  ({a.clause})");
            sb.AppendLine("Press R to walk it again.");
            Debug.Log(sb.ToString());
        }

        void Restart()
        {
            _flagged.Clear();
            _log.Clear();
            _finished = false;
            UnityEngine.SceneManagement.SceneManager.LoadScene(
                UnityEngine.SceneManagement.SceneManager.GetActiveScene().buildIndex);
        }

        void OnGUI()
        {
            var key = importer != null ? importer.AnswerKey.ToList() : new List<AnswerEntry>();
            int found = key.Count(a => _flagged.Contains(a.entityId));

            GUI.Box(new Rect(10, 10, 320, 68), "");
            GUI.Label(new Rect(20, 16, 300, 20),
                importer != null && importer.Contract != null ? importer.Contract.scenarioId : "");
            GUI.Label(new Rect(20, 36, 300, 20), $"Hazards found: {found} / {key.Count}");
            GUI.Label(new Rect(20, 54, 300, 20),
                _finished ? "Finished. R to restart." : "WASD move, right-drag look, click to flag, Enter to finish");

            // crosshair
            GUI.Label(new Rect(Screen.width / 2f - 4, Screen.height / 2f - 10, 20, 20), "+");
        }
    }
}
