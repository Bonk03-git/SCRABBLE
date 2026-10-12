# SCRABBLE
A system for analyzing the progress of a Scrabble game in real time; it reads the board, the tiles on it, and the scores for each player.
It works similarly to the feature offered by chess.com during chess tournaments.

## Creating labels (marking the board on photos)

The labeling tool builds the photo dataset. On each photo you mark the 4 corners of the board by hand. The tool saves their coordinates and moves the photo to the annotated folder.

### Running it

1. Put the board photos (JPG or PNG) in `dataset/photos/`.
2. From the project folder, run:

   ```bash
   ./start
   ```

   On the first run, the script creates a Python 3.11 environment and installs the packages.

### Point order: THE MOST IMPORTANT RULE

> **Always click the points in the same order:
> 1 = TOP-LEFT corner, then CLOCKWISE.**

```
   1 ───────────── 2
   │               │
   │     BOARD     │
   │               │
   4 ───────────── 3
```

| Point | Corner |
|:---:|---|
| **1** | top-left |
| **2** | top-right |
| **3** | bottom-right |
| **4** | bottom-left |

Why this matters:
- **The tool saves the points exactly in the order you click them** and never reorders them. The order in the label file is the order you clicked.
- This way a label records the **orientation of the board**. If the board is rotated in the photo, point 1 is still the top-left corner **of the board**, even if it lands on, say, the right side of the photo.
- Clicking in the wrong order produces a wrong label. A future model would learn the wrong board orientation from it, and that mistake can't be detected automatically later.

What the tool checks for you:
- Next to each point it shows its number, and the bar at the top tells you which corner to click next, e.g. `click 2: top-right`.
- After 4 clicks, the outline is **green** if the points go clockwise. It is **red** if they go counter-clockwise or the sides cross, and saving is then blocked.
- The tool **can't tell** which corner of the board really is the top-left. Green only means the direction is right. You have to make sure you started from the correct corner.

### What to mark

Click the **outer corners of the 15×15 cell grid**, i.e. the corners of the outermost playing cells. Don't click the corners of the plastic frame or the box. For precise clicks, use the loupe: a 5× zoom around the cursor with a red crosshair. It always sits on the opposite side of the window from the cursor.

### Keys

| Key | Action |
|---|---|
| left click | add the next point (1 → 2 → 3 → 4) |
| right click / `u` / Backspace | undo the last point |
| Enter / Space | save and go to the next photo |
| `d` / `a` | next / previous photo without saving |
| `q` / Esc | quit |

### What happens after saving

```
dataset/
├── photos/      photos waiting to be annotated
├── annotated/   annotated photos (the tool moves them here after saving)
└── labels/      one .json file per annotated photo
```

A label has the same name as its photo, e.g. `annotated/abc.jpeg` → `labels/abc.json`:

```json
{
  "image": "abc.jpeg",
  "corners": [[294.4, 455.7], [1231.4, 437.8], [1254.4, 1425.9], [271.4, 1431.0]]
}
```

- `corners` holds points 1, 2, 3, 4 in that order: top-left, top-right, bottom-right, bottom-left.
- The `[x, y]` coordinates are in pixels of the original photo: `x` from the left, `y` from the top. Large photos are scaled down for display only.
- If `annotated/` already has a photo with the same name, the tool won't overwrite it. It shows a message and leaves the photo in `photos/`.

### Fixing labels

```bash
./start dataset/annotated
```

The tool loads the existing points. Undo them with right click and click them again. After saving, the label is overwritten and the photo stays in `annotated/`.
