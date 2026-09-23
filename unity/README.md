# Unity runtime

The scene contract (`scene_unity.json`) is engine agnostic. This folder holds a
reference importer that consumes it in Unity; `safesitegen/environment.py` holds
a browser renderer that consumes exactly the same file. Neither is privileged.

The importer never re-decides what counts as a hazard. That was settled by the
validation gate before the file was written, and the importer refuses to build a
scene whose `validated` flag is false.

---

## Setup, about ten minutes

1. **Install Unity Hub** and any **Unity 2021.3 LTS or newer** editor. Both the
   Built-in and URP render pipelines work; the importer only uses primitives and
   standard materials.
2. **New project** → 3D → name it `SafeSiteGenDemo`.
3. In the Project panel create `Assets/Scripts/` and drag in both files from
   `unity/Assets/Scripts/`:
   - `SafeSiteGenImporter.cs`
   - `TraineeController.cs`
4. Create `Assets/StreamingAssets/` (exact spelling, it is a special folder) and
   drop in a generated `scene_unity.json`:

   ```bash
   python -m safesitegen prompt "ironworker on a steel deck with an unprotected \
       leading edge and a crane swinging a load near the crew" --out demo/
   cp demo/scene_unity.json /path/to/SafeSiteGenDemo/Assets/StreamingAssets/
   ```

5. In the Hierarchy: right click → Create Empty, name it `SafeSiteGen`, and add
   the **SafeSiteGenImporter** component in the Inspector.
6. Press **Play**.

You should see the site build from primitives, the trainee route drawn as a
line, and a HUD reading `Hazards found: 0 / 3`. Right-drag to look, WASD to
move, left click to flag, Enter to score.

The Console prints the scenario id, the trade, the difficulty and the number of
teaching points on load, and one line per flag with the clause reference.

---

## Swapping in real assets

Leave the code alone. In the Inspector, expand **Prefab Bindings**, set the size,
and for each row give an `assetType` string and a prefab:

| assetType | typical prefab |
|---|---|
| `guardrail_system` | a guardrail section |
| `portable_ladder` | an extension ladder |
| `mobile_crane` | a crane |
| `haul_truck` | a dump truck |
| `spoil_pile` | a dirt mound |
| `rebar_cage` | protruding starter bars |
| `ironworker`, `labourer`, `pipelayer` | a rigged worker character |

Anything without a binding falls back to a primitive, so a partial asset library
still produces a complete scene. Asset types come straight from the taxonomy;
`python -m safesitegen classes` lists them.

---

## Troubleshooting

**Nothing appears and the Console says the contract was not found.**
`StreamingAssets` must sit directly under `Assets`, spelled exactly that way.
Check the path Unity prints.

**"Refusing to build it as a training environment."**
The JSON came from a scenario the gate rejected. Regenerate without `--no-gate`.

**Everything is a grey box.**
Expected with no prefab bindings. The colours are by entity kind: amber workers,
blue equipment, green controls, purple environment, grey structure.

**Clicking does nothing.**
Left click flags, right-drag looks. If you are holding right mouse while
clicking, the raycast still fires but you may be aimed away. The crosshair is
the `+` at screen centre.

**Hazards and distractors look identical.**
That is the design. A compliant guardrail at 42 inches and a defective one at 33
inches are the same `assetType`, so the trainee has to judge the condition rather
than recognise an object. Same reason the checker evaluates the clause instead of
matching a label.

**`JsonUtility` throws on the site block.**
Unity's `JsonUtility` cannot deserialise jagged arrays, which is why `patrol` is
emitted as objects with `x` and `y` rather than nested arrays. If you have edited
the exporter, keep that shape. It also materialises a JSON `null` as an empty
object rather than null, which is why `HazardMarker.IsHazard` tests for an empty
`hazardId` instead of a null reference.

---

## What this is and is not

It is a faithful consumer of the runtime contract, enough to show that a prompt
produces a walkable, scoreable training environment, and enough to swap real
assets into without code changes.

It is not a finished training product. No XR rig, no navmesh, no animation, no
gaze telemetry, no physics beyond colliders for raycasting. Those are runtime
engineering, and the proposal treats them as such: the research contribution is
upstream, in whether the content that reaches the trainee is verifiably correct.
