with open(UNLEARN_JSON, "r") as handle:
    coco = json.load(handle)
image_by_id = {item["id"]: item for item in coco["images"]}
poison_annotations = {}
for annotation in coco["annotations"]:
    filename = image_by_id[annotation["image_id"]]["file_name"]
    poison_annotations.setdefault(filename, []).append(list(annotation["bbox"]))

unlearn_files = sorted(Path(UNLEARN_DIR).glob("*.png"))
file_by_name = {path.name: path for path in unlearn_files}
source_ids = sorted(name for name in poison_annotations if name in file_by_name)
assert len(source_ids) >= CFG["N_FOLDS"] * 2, "too few poison sources for source-disjoint folds"

def inpaint_boxes(gray, boxes_xywh, padding=8):
    output = gray.copy()
    for x, y, width, height in boxes_xywh:
        x1 = int(max(0, x - padding))
        y1 = int(max(0, y - padding))
        x2 = int(min(gray.shape[1], x + width + padding))
        y2 = int(min(gray.shape[0], y + height + padding))
        ring_x1 = max(0, x1 - 32)
        ring_y1 = max(0, y1 - 32)
        ring_x2 = min(gray.shape[1], x2 + 32)
        ring_y2 = min(gray.shape[0], y2 + 32)
        ring = gray[ring_y1:ring_y2, ring_x1:ring_x2]
        median = float(np.median(ring))
        sigma = 1.4826 * float(np.median(np.abs(ring - median))) + 1e-3
        local_rng = np.random.default_rng(
            int(hashlib.sha1(f"{x1}:{y1}:{x2}:{y2}".encode()).hexdigest()[:8], 16)
        )
        output[y1:y2, x1:x2] = np.clip(
            local_rng.normal(median, sigma, (y2 - y1, x2 - x1)), 0, 255
        ).astype(np.float32)
    return output

source_records = {}
for source_id in source_ids:
    gray = load_gray(file_by_name[source_id])
    background = inpaint_boxes(gray, poison_annotations[source_id])
    patches = []
    for x, y, width, height in poison_annotations[source_id]:
        padding = 10
        x1 = int(max(0, x - padding))
        y1 = int(max(0, y - padding))
        x2 = int(min(gray.shape[1], x + width + padding))
        y2 = int(min(gray.shape[0], y + height + padding))
        crop = gray[y1:y2, x1:x2].copy()
        signal = np.clip(crop - float(np.median(crop)), 0, None)
        inner_box = np.array([x - x1, y - y1, x + width - x1, y + height - y1], np.float32)
        patches.append({"signal": signal, "inner_box": inner_box})
    source_records[source_id] = {
        "path": file_by_name[source_id],
        "background": background,
        "patches": patches,
    }

def split_source_folds(identities, number_of_folds, seed):
    ordered = sorted(
        identities,
        key=lambda value: hashlib.sha1(f"{seed}:{value}".encode()).hexdigest(),
    )
    folds = [[] for _ in range(number_of_folds)]
    for index, identity in enumerate(ordered):
        folds[index % number_of_folds].append(identity)
    assert sorted(sum(folds, [])) == sorted(identities)
    assert all(folds), "every source fold must be non-empty"
    return folds

SOURCE_FOLDS = split_source_folds(source_ids, CFG["N_FOLDS"], CFG["SEED"])
for fold_index, fold_sources in enumerate(SOURCE_FOLDS):
    print(f"source fold {fold_index}:", fold_sources)

def random_background(allowed_sources, rng):
    source_id = allowed_sources[int(rng.integers(0, len(allowed_sources)))]
    background = source_records[source_id]["background"]
    rotation = int(rng.integers(0, 4))
    transformed = np.rot90(background, rotation)
    if rng.random() < 0.5:
        transformed = transformed[:, ::-1]
    return np.ascontiguousarray(transformed.copy())

def make_matched_pair(allowed_sources, rng, forced_source=None):
    source_id = forced_source
    if source_id is None:
        source_id = allowed_sources[int(rng.integers(0, len(allowed_sources)))]
    assert source_id in allowed_sources
    patch_record = source_records[source_id]["patches"][
        int(rng.integers(0, len(source_records[source_id]["patches"])))
    ]
    signal = patch_record["signal"].copy()
    inner = patch_record["inner_box"].copy()
    if rng.random() < 0.5:
        signal = signal[:, ::-1]
        old_x1, old_x2 = inner[0], inner[2]
        inner[0] = signal.shape[1] - old_x2
        inner[2] = signal.shape[1] - old_x1
    scale = float(rng.uniform(0.90, 1.10))
    new_width = max(8, int(round(signal.shape[1] * scale)))
    new_height = max(8, int(round(signal.shape[0] * scale)))
    signal = cv2.resize(signal, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    inner *= scale
    control = random_background(allowed_sources, rng)
    image_height, image_width = control.shape
    assert new_height < image_height - 4 and new_width < image_width - 4
    x0 = int(rng.integers(2, image_width - new_width - 2))
    y0 = int(rng.integers(2, image_height - new_height - 2))
    poison = control.copy()
    strength = float(rng.uniform(0.85, 1.15))
    poison[y0:y0 + new_height, x0:x0 + new_width] = np.clip(
        poison[y0:y0 + new_height, x0:x0 + new_width] + strength * signal,
        0,
        255,
    )
    target = inner + np.array([x0, y0, x0, y0], np.float32)
    return {
        "poison": to_three(poison),
        "control": to_three(control),
        "target": target[None, :],
        "source_id": source_id,
    }

def make_validation_pairs(allowed_sources, fold_index, count=32, seed_offset=0):
    rng = np.random.default_rng(10000 + 100 * fold_index + seed_offset)
    pair_count = max(count, len(allowed_sources))
    pairs = [
        make_matched_pair(
            allowed_sources,
            rng,
            forced_source=allowed_sources[index % len(allowed_sources)],
        )
        for index in range(pair_count)
    ]
    assert set(pair["source_id"] for pair in pairs) == set(allowed_sources)
    return pairs

print("poison sources:", len(source_ids), "| patches:", sum(len(v["patches"]) for v in source_records.values()))
