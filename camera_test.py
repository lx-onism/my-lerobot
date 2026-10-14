import cv2, time

for name, idx in [("wrist", 0), ("front", 1)]:
    for backend, bname in [(cv2.CAP_ANY, "ANY"), (cv2.CAP_DSHOW, "DSHOW")]:
        cap = cv2.VideoCapture(idx, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(30): cap.read()          # 预热
        t0 = time.time()
        n = 100
        for _ in range(n): cap.read()
        dt = (time.time() - t0) / n * 1000
        print(f"{name} [{bname}]  {dt:.1f} ms/帧  实际FPS={1000/dt:.1f}")
        cap.release()