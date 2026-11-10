import numpy as np
from scipy.optimize import least_squares
# (image x, image y) points on lines; world: X across (photo left->right), Y along (photo top->bottom), W = 1
L = {
 ("X", "0"):    [(61.5,140),(56,160),(49.5,180),(43.5,200),(37.5,220),(31,240),(25,260),(19,280),(13.5,300),(9.5,320)],
 ("X", "1"):    [(412,100),(419.5,120),(427.5,140),(435,160),(442.5,180),(450,200),(458,220),(465.5,240),(473,260)],
 ("Y", "0"):    [(180,19),(200,19),(220,19),(240,19),(260,19),(280,19),(300,19.5)],
 ("Y", "D"):    [(160,470),(180,470),(200,469),(220,468.5),(240,468),(260,467.5),(280,467.5),(300,467),(320,467.5),(340,467.5),(360,467),(380,466.5)],
 ("X", "c"):    [(68.5,240),(63.5,260),(58.5,280),(54,300),(50,320)],
 ("X", "1-c"):  [(375,80),(378,90),(380.5,100),(383.5,110),(386.5,120),(420,220),(426.5,240),(433,260),(438,280),(445,300),(450,320)],
 ("Y", "c"):    [(180,32),(200,32),(220,32),(240,32.5),(260,32.5),(280,33),(300,33.5)],
 ("Y", "D-c"):  [(160,417.5),(180,417),(200,416),(220,416.5),(240,416.5),(260,417),(280,417),(300,416.5),(320,416.5),(340,416.5),(360,415.5),(380,415.5)],
 ("X", "c+e"):  [(143,70),(141.5,80),(139.5,90),(137.5,100),(135.5,110),(133.5,120),(108,260),(104.5,280),(101,300),(97,320)],
 ("X", "1-c-e"):[(344,70),(346.5,80),(349,90),(351.5,100),(354,110),(356.5,120),(386.5,240),(391.5,260),(396.5,280),(401.5,300),(406,320)],
 ("Y", "c+e"):  [(180,47.5),(200,47.5),(220,47.5),(240,47.5),(260,47.5),(280,47.5),(300,47.5)],
 ("Y", "D-c-e"):[(200,370),(220,369),(240,369),(260,368),(280,367.5),(300,367.5),(320,367),(340,367),(360,367),(380,367)],
}
def target(expr, D, c, e):
    return eval(expr, {}, {"D": D, "c": c, "e": e})
def unpack(p):
    H = np.append(p[:8], 1.0).reshape(3, 3); return H, p[8], p[9], p[10]
def to_world(Hinv, pts):
    P = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Hinv.T
    return P[:, :2] / P[:, 2:]
def resid(p):
    H, D, c, e = unpack(p); Hinv = np.linalg.inv(H); r = []
    for (axis, expr), pts in L.items():
        w = to_world(Hinv, pts); k = target(expr, D, c, e)
        r += list((w[:, 0] if axis == "X" else w[:, 1]) - k)
    return np.array(r)
# init: world->image homography from rough corners
import cv2
D0 = 1.4
src = np.float32([[0,0],[1,0],[1,D0],[0,D0]]); dst = np.float32([[95,19],[400,19],[480,468],[0,470]])
H0 = cv2.getPerspectiveTransform(src, dst); H0 /= H0[2,2]
p0 = np.r_[H0.ravel()[:8], D0, 0.12, 0.12]
sol = least_squares(resid, p0, x_scale="jac")
H, D, c, e = unpack(sol.x)
print("rms world residual (W units):", np.sqrt(np.mean(sol.fun**2)).round(4))
print(f"aspect D/W = {D:.3f}, outer lane c/W = {c:.4f}, inner lane e/W = {e:.4f}")
np.save("H.npy", H); np.save("params.npy", np.array([D, c, e]))
Hinv = np.linalg.inv(H)
def w(pts): return to_world(Hinv, pts)
# connector: upper island bottom / centre / lower island top, at photo x=240
print("connector lines Y:", w([(240,142.5),(240,165.5),(240,190.5)])[:,1].round(4))
# junction gaps: middle line ends near junction (left: middle-left upper end ~ (98?,125) lower start (70,215)?)
# cars (image px): connector car body (172..210 at y~157), lower-left car width (76..110 at y~243)
cl = w([(172,157),(210,157)]); print("connector car length (W units):", np.linalg.norm(cl[1]-cl[0]).round(4))
cw = w([(76,243),(110,243)]); print("lower-left car width (W units):", np.linalg.norm(cw[1]-cw[0]).round(4))
