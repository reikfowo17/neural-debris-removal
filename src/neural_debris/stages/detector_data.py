BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1

CANDIDATE_ROOTS = [
    "/kaggle/input/neural-debris-removal-in-streak-detection-models",
    "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models",
]
DATA_ROOT = next((root for root in CANDIDATE_ROOTS if os.path.isdir(root)), None)
if DATA_ROOT is None:
    annotation_hits = glob.glob("/kaggle/input/**/annotations_coco.json", recursive=True)
    assert annotation_hits, "competition data not found under /kaggle/input"
    DATA_ROOT = str(Path(annotation_hits[0]).parents[1])

weight_hits = sorted(glob.glob(str(Path(DATA_ROOT) / "**" / "*.pth"), recursive=True))
assert weight_hits, "no model weights found"
POISONED_WEIGHTS = next((path for path in weight_hits if "poison" in path.lower()), weight_hits[0])
UNLEARN_JSON = sorted(glob.glob(str(Path(DATA_ROOT) / "**" / "annotations_coco.json"), recursive=True))[0]
UNLEARN_DIR = str(Path(UNLEARN_JSON).parent)
png_counts = {}
for png_path in glob.glob(str(Path(DATA_ROOT) / "**" / "*.png"), recursive=True):
    parent = str(Path(png_path).parent)
    png_counts[parent] = png_counts.get(parent, 0) + 1
TEST_DIR = max(png_counts, key=png_counts.get)
SAMPLE_SUB = sorted(glob.glob(str(Path(DATA_ROOT) / "**" / "sample_submission.csv"), recursive=True))[0]

def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, path
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    else:
        image = image.astype(np.float32) / 255.0
    image = np.clip(image * 255.0, 0, 255).astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return image

def load_gray(path):
    return load_image(path)[:, :, 0].copy()

def to_three(gray):
    return np.repeat(gray[:, :, None], 3, axis=2)

def iou_matrix(first, second):
    first = np.asarray(first, np.float32).reshape(-1, 4)
    second = np.asarray(second, np.float32).reshape(-1, 4)
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), np.float32)
    left = np.maximum(first[:, None, :2], second[None, :, :2])
    right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    size = np.maximum(0.0, right - left)
    intersection = size[:, :, 0] * size[:, :, 1]
    area_first = np.prod(np.maximum(0.0, first[:, 2:] - first[:, :2]), axis=1)[:, None]
    area_second = np.prod(np.maximum(0.0, second[:, 2:] - second[:, :2]), axis=1)[None, :]
    union = area_first + area_second - intersection
    return np.where(union > 0, intersection / union, 0.0).astype(np.float32)

def build_cfg(weights, score_threshold):
    config = get_cfg()
    config.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    config.MODEL.WEIGHTS = str(weights)
    config.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    config.MODEL.RETINANET.SCORE_THRESH_TEST = score_threshold
    config.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    config.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    config.MODEL.DEVICE = DEVICE
    return config

def load_model(weights=POISONED_WEIGHTS):
    model = build_model(build_cfg(weights, CFG["CAND_THRESH"]))
    DetectionCheckpointer(model).load(str(weights))
    model.to(DEVICE)
    model.eval()
    return model

resize_aug = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)

def to_batched(image, boxes_xyxy=None, train=False):
    height, width = image.shape[:2]
    tfm = resize_aug.get_transform(image)
    resized = tfm.apply_image(image)
    record = {
        "image": torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1))),
        "height": height,
        "width": width,
    }
    transformed_boxes = np.zeros((0, 4), np.float32)
    if boxes_xyxy is not None and len(boxes_xyxy):
        transformed_boxes = tfm.apply_box(np.asarray(boxes_xyxy, np.float32))
    record["anchor_boxes_resized"] = torch.as_tensor(transformed_boxes, dtype=torch.float32)
    if train:
        instances = Instances(resized.shape[:2])
        instances.gt_boxes = Boxes(torch.as_tensor(transformed_boxes, dtype=torch.float32))
        instances.gt_classes = torch.zeros((len(transformed_boxes),), dtype=torch.int64)
        record["instances"] = instances
    return record

@torch.no_grad()
def infer_np(model, image):
    was_training = model.training
    model.eval()
    with torch.autocast("cuda", enabled=(CFG["AMP_INFER"] and DEVICE == "cuda")):
        instances = model([to_batched(image, train=False)])[0]["instances"].to("cpu")
    if was_training:
        model.train()
    return (
        instances.pred_boxes.tensor.numpy().astype(np.float32),
        instances.scores.numpy().astype(np.float32),
    )

def permute_head_output(tensor, channels_per_anchor):
    batch_size, combined_channels, height, width = tensor.shape
    assert combined_channels % channels_per_anchor == 0
    anchor_count = combined_channels // channels_per_anchor
    return tensor.view(
        batch_size, anchor_count, channels_per_anchor, height, width
    ).permute(0, 3, 4, 1, 2).reshape(batch_size, -1, channels_per_anchor)

def head_tensors(model, batched):
    images = model.preprocess_image(batched)
    feature_dict = model.backbone(images.tensor)
    features = [feature_dict[name] for name in model.head_in_features]
    logits_by_level, deltas_by_level = model.head(features)
    logits = torch.cat([permute_head_output(value, NUM_CLASSES) for value in logits_by_level], dim=1)
    deltas = torch.cat([permute_head_output(value, 4) for value in deltas_by_level], dim=1)
    generated_anchors = model.anchor_generator(features)
    anchor_levels = generated_anchors
    anchors = Boxes.cat(anchor_levels).tensor
    return logits, deltas, anchors

print("data root:", DATA_ROOT)
print("weights  :", POISONED_WEIGHTS)
print("unlearn  :", UNLEARN_DIR)
print("test     :", TEST_DIR, png_counts[TEST_DIR])
