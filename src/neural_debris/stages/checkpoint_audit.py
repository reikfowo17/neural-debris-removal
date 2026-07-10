def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

TRAINING_GATES_PASS = False
if ALL_GATES_PASS:
    TRAINING_GATES_PASS = True
    CHECKPOINT_FINGERPRINTS = {
        Path(path).name: sha256_file(path)
        for path in [WORK / "nb51_legacy_ranker.pth", *fold_checkpoints]
    }
    print("NB51 training gates passed; starting fixed-alpha OOF audit.")
else:
    raise RuntimeError(
        "NB51 FAIL CLOSED: standalone training did not pass every hard gate."
    )
