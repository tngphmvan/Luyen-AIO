import cv2
import os

frame_dir = "frames/"
frames = sorted(os.listdir(frame_dir))
first = cv2.imread(os.path.join(frame_dir, frames[0]))
h, w, _ = first.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (w, h))

for f in frames:
    img = cv2.imread(os.path.join(frame_dir, f))
    cv2.imshow("Playback", img)
    out.write(img)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
