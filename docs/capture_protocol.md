# Capture protocol (one page)

Bring **one** of the three captures. Before starting: **all lights on, every interior door fully open, curtains open.**

| Tier | Phone | Install |
|---|---|---|
| **LiDAR** | iPhone 15 Pro / Pro Max or newer Pro (black LiDAR dot by the cameras) | **Stray Scanner** (free) |
| **Video** | any iPhone 15 or newer | **NeRFCapture** (free) |
| **Photos** | any iPhone 15 or newer | nothing: built-in Camera |

## How to walk (LiDAR and Video)
1. Start in the first room, about 1 m from a wall. Hold the phone upright at chest height. Walk at **half** normal speed.
2. In every room, **sweep each wall from the floor up to the ceiling and back**, and point at the ceiling once.
   *Walls hidden behind furniture and ceilings you never point at come out "not observed".*
3. **Step into every room and walk through every doorway.** Rooms only seen from outside are left out.
4. One continuous capture for the whole home. **Finish where you started.** 2–4 minutes for an apartment.

**LiDAR:** open Stray Scanner and tap the red **Record**, walk as above, then tap **Stop**.
Hand-off: Files → On My iPhone → Stray Scanner → newest folder → Share → AirDrop to the Mac.

**Video:** open NeRFCapture, choose **Offline**, tap **Start**. Walk as above and tap capture **every half step and
every ~20° of turning** (80–150 captures), holding still for each one. Tap **End**.
Hand-off: Files → On My iPhone → NeRFCapture → newest zip → AirDrop to the Mac.
*(No NeRFCapture? Record the same walk as a normal Camera video, 1x, landscape. It runs, but is much less accurate.)*

## Photos (Camera app)
1. **Photo** mode, **1x** lens, flash off, phone **landscape** and level. On the Mac, make **one folder per room**.
2. **Each room, 4 photos:** stand in the middle, one photo every quarter turn, each overlapping the last by a third.
   Every photo shows some floor and the ceiling line.
3. **Each doorway, 2 photos:** stand *in* the doorway, one photo into each room. Put **copies of both** in **both**
   rooms' folders. This is how rooms are joined.
4. Each folder ends with **2–8** photos. If it goes over 8, drop room photos and keep the doorway photos.
5. Put all room folders in one parent folder.

## Avoid
- Sweeping slowly across mirrors, glass screens and windows: glance past them.
- Dark rooms: switch a light on. People or pets walking through.
- Turning fast: a full turn should take at least 5 seconds.

## Run it (one command)
`scan2plan run <Stray Scanner folder | NeRFCapture folder or zip | photo parent folder | clip.mov>`
Output: `out/<name>/<tier>/plan.json`, `plan.png`, `plan.svg`.
