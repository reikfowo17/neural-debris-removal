def add_clean_streaks(gray, rng):
    output = gray.copy()
    median = float(np.median(output))
    noise = 1.4826 * float(np.median(np.abs(output - median))) + 1e-3
    boxes = []
    for _ in range(int(rng.integers(1, 3))):
        length = float(rng.uniform(24.0, 90.0))
        angle = float(rng.uniform(0, np.pi))
        center_x = float(rng.uniform(80, output.shape[1] - 80))
        center_y = float(rng.uniform(80, output.shape[0] - 80))
        dx = 0.5 * length * np.cos(angle)
        dy = 0.5 * length * np.sin(angle)
        x1, y1 = center_x - dx, center_y - dy
        x2, y2 = center_x + dx, center_y + dy
        sigma = float(rng.uniform(1.2, 3.2))
        snr = float(rng.uniform(4.0, 30.0))
        canvas = np.zeros_like(output, dtype=np.float32)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, 1, cv2.LINE_AA)
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigma)
        if canvas.max() > 0:
            canvas /= canvas.max()
        output = np.clip(output + canvas * snr * noise, 0, 255)
        boxes.append([
            max(0.0, min(x1, x2) - 3 * sigma),
            max(0.0, min(y1, y2) - 3 * sigma),
            min(float(output.shape[1]), max(x1, x2) + 3 * sigma),
            min(float(output.shape[0]), max(y1, y2) + 3 * sigma),
        ])
    return output, np.asarray(boxes, np.float32)

def draw_legacy_forget(rng):
    draw = float(rng.random())
    if draw < 0.15:
        source_id = source_ids[int(rng.integers(0, len(source_ids)))]
        original = load_gray(source_records[source_id]["path"])
        return to_three(original), np.zeros((0, 4), np.float32)
    pair = make_matched_pair(source_ids, rng)
    if draw < 0.40:
        return pair["poison"], np.zeros((0, 4), np.float32)
    gray, clean_boxes = add_clean_streaks(pair["poison"][:, :, 0], rng)
    return to_three(gray), clean_boxes

def draw_legacy_retain(rng):
    gray = random_background(source_ids, rng)
    if rng.random() < 2.0 / 3.0:
        gray, _ = add_clean_streaks(gray, rng)
    return to_three(gray)
