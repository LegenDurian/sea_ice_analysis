import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

class FloeSeparator:
    """This class is used to process a singular thermal sea ice image into segmented ice floes. 
    Results are stored as self.labels_filtered
    """
    def __init__(self, img_path):
        # load image
        self.img_path = img_path
        self.img = cv.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(f"Could not read image at: {os.path.abspath(img_path)}")

        self.img_gray = None
        self.img_rgb = None
        self.mask = None
        self.labels_filtered = None
        self.areas_px = None
        self.centroids = None

    @staticmethod
    def save_rgb_as_bgr(path, img_rgb):
        """Accepts float/int arrays, converts to uint8 BGR, then writes."""
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
        cv.imwrite(path, img_bgr)
        return

    def preprocess(self, img_path=None, clahe_clip=1.5, clahe_grid=(8, 8), thresh_val=70):
        """Load image, CLAHE-equalize gray, binary-inv threshold. Returns (img_gray, img_rgb, mask)."""
        # if a new path is provided, update self.img
        if img_path is not None:
            self.img_path = img_path
            self.img = cv.imread(img_path)

        img = self.img
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {os.path.abspath(self.img_path)}")

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_rgb  = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # CLAHE
        clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        img_gray_eq = clahe.apply(img_gray)
        cv.imwrite('clahe_output.jpg', img_gray_eq)

        # Binary INV: 255=ice, 0=sea
        _, mask = cv.threshold(img_gray_eq, thresh_val, 255, cv.THRESH_BINARY_INV)
        mask = mask.astype(np.uint8)

        # store on self so they are accessible later
        self.img_gray = img_gray
        self.img_rgb = img_rgb
        self.mask = mask

        return img_gray, img_rgb, mask

    def split_erode_dilate(self, mask, min_area=30):
        k3 = np.ones((3,3), np.uint8)
        eroded = cv.erode(mask, k3, iterations=1)
        num_labels, labels = cv.connectedComponents(eroded)
        # Dilate each label back separately, then stack (keeps identities)
        labels_out = np.zeros_like(labels, dtype=np.int32)
        nid = 1
        for lbl in range(1, num_labels):
            r = (labels == lbl).astype(np.uint8) * 255
            d = cv.dilate(r, k3, iterations=1)
            # Constrain by original mask to avoid leaking into sea
            d = cv.bitwise_and(d, mask)
            a = int((d > 0).sum())
            if a >= min_area:
                labels_out[d > 0] = nid
                nid += 1
        # areas/centroids
        areas_px, centroids = [], []
        for i in range(1, nid):
            rr = (labels_out == i)
            areas_px.append(int(rr.sum()))
            ys, xs = np.nonzero(rr)
            centroids.append((float(xs.mean()), float(ys.mean())))

        # store on self
        self.labels_filtered = labels_out
        self.areas_px = areas_px
        self.centroids = centroids

        return labels_out, areas_px, centroids

    def split_erode_dilate_small(self, mask, min_area=30, erode_min_size=500):
        k3 = np.ones((3,3), np.uint8)

        # Work from original mask components first
        num0, labels0, stats0, cents0 = cv.connectedComponentsWithStats(mask, connectivity=8, ltype=cv.CV_32S)

        labels_out = np.zeros_like(labels0, dtype=np.int32)
        areas_px, centroids = [], []
        nid = 1

        for i in range(1, num0):
            area0 = int(stats0[i, cv.CC_STAT_AREA])
            if area0 < min_area:
                continue

            comp = (labels0 == i).astype(np.uint8) * 255

            if area0 < erode_min_size:
                rr = (comp > 0)
                labels_out[rr] = nid
                areas_px.append(int(rr.sum()))
                ys, xs = np.nonzero(rr)
                centroids.append((float(xs.mean()), float(ys.mean())))
                nid += 1
                continue

            # Erode/dilate only for large components
            eroded = cv.erode(comp, k3, iterations=1)
            num_labels, labels = cv.connectedComponents(eroded)
            # Dilate each label back separately, then stack (keeps identities)
            for lbl in range(1, num_labels):
                r = (labels == lbl).astype(np.uint8) * 255
                d = cv.dilate(r, k3, iterations=1)
                # Constrain by original mask to avoid leaking into sea
                d = cv.bitwise_and(d, comp)
                a = int((d > 0).sum())
                if a >= min_area:
                    labels_out[d > 0] = nid
                    ys, xs = np.nonzero(d)
                    centroids.append((float(xs.mean()), float(ys.mean())))
                    areas_px.append(a)
                    nid += 1

        # store on self
        self.labels_filtered = labels_out
        self.areas_px = areas_px
        self.centroids = centroids

        return labels_out, areas_px, centroids

    def split_simple(self, mask, min_area=30):
        """
        Simple connected-components segmentation directly on the binary mask.
        No erosion, no watershed.
        - mask: 255=ice, 0=sea
        - min_area: filter out tiny specks
        Returns (labels_filtered, areas_px, centroids)
        """
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv.CV_32S
        )
        labels_filtered = np.zeros_like(labels, dtype=np.int32)
        areas_px = []
        centroids_new = []
        next_id = 1
        for i in range(1, num_labels):  # skip background (label 0)
            area_i = int(stats[i, cv.CC_STAT_AREA])
            if area_i >= min_area:
                labels_filtered[labels == i] = next_id
                areas_px.append(area_i)
                centroids_new.append(tuple(centroids[i]))  # (cx, cy)
                next_id += 1

        # store on self
        self.labels_filtered = labels_filtered
        self.areas_px = areas_px
        self.centroids = centroids_new

        return labels_filtered, areas_px, centroids_new

    def get_colored_segments(self, labels_filtered, count, color_seed=0):
        """
        Given filtered labels and count (number of distinct labels), produce a colorized segmentation image.
        Returns image of segmented regions in random colors.
        """
        if count > 0:
            rng = np.random.default_rng(color_seed)
            lut = np.vstack([[0, 0, 0], rng.integers(0, 255, size=(count, 3), dtype=np.uint8)])
            seg_rgb = lut[labels_filtered].astype(np.uint8)
        else:
            seg_rgb = np.zeros((*labels_filtered.shape, 3), dtype=np.uint8)
        return seg_rgb

    def get_numbered_segments(self, img_gray, labels_filtered, centroids, count):
        """
        Given grayscale image, filtered labels, centroids, and count,
        produce an overlay image with red boundaries and numbers at centroids.
        """
        overlay_rgb = cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)
        for lbl in range(1, count + 1):
            comp_mask = (labels_filtered == lbl).astype(np.uint8) * 255
            if comp_mask.sum() == 0:
                continue
            contours, _ = cv.findContours(comp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(overlay_rgb, contours, -1, (255, 0, 0), 1, lineType=cv.LINE_AA)  # red borders

        # Numbers at centroids
        overlay_bgr = cv.cvtColor(overlay_rgb, cv.COLOR_RGB2BGR)
        H, W = overlay_bgr.shape[:2]
        font_scale = max(0.5, min(2.0, H / 1000.0))
        for idx, (cx, cy) in enumerate(centroids, start=1):
            x, y = int(round(cx)), int(round(cy))
            cv.putText(overlay_bgr, str(idx), (x, y),
                    cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv.LINE_AA)  # outline
            cv.putText(overlay_bgr, str(idx), (x, y),
                    cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)  # fill
        overlay_numbered = cv.cvtColor(overlay_bgr, cv.COLOR_BGR2RGB)
        return overlay_numbered

    def visualize_and_save(self, img_gray, img_rgb, mask, labels_filtered, centroids, areas_px,
                        color_seed=0, hist_path='size_distribution_hist.png'):
        """
        Make a compact 2x2 figure:
        1) Original grayscale
        2) Binary mask
        3) Segmentation (random color per label)
        4) Grayscale overlaid with red boundaries + numbers
        """
        count = int(labels_filtered.max())
        print("Estimated number of ice chunks:", count)

        seg_rgb = self.get_colored_segments(labels_filtered, count, color_seed)

        overlay_numbered = self.get_numbered_segments(img_gray, labels_filtered, centroids, count)

        # figure
        plt.figure(figsize=(12, 9))
        plt.subplot(221)
        plt.title("Original (gray)")
        plt.imshow(img_gray, cmap='gray')
        plt.axis('off')

        plt.subplot(222)
        plt.title("Binary mask (ice=white, sea=black)")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(223)
        plt.title(f"Segmentation (N={count})")
        plt.imshow(seg_rgb)
        plt.axis('off')

        plt.subplot(224)
        plt.title("Overlay: red edges + IDs")
        plt.imshow(overlay_numbered)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('figure_2x2.png', dpi=150)
        plt.show()

        # size distribution
        if len(areas_px) > 0:
            areas_px = np.asarray(areas_px, dtype=np.int32)
            print(f"Area stats (pixels^2) — N={len(areas_px)}")
            print(f"  min: {areas_px.min():,}")
            print(f"  max: {areas_px.max():,}")
            print(f"  mean: {areas_px.mean():,.2f}")
            print(f"  median: {np.median(areas_px):,.2f}")

            plt.figure(figsize=(8, 5))
            plt.hist(areas_px, bins='auto', edgecolor='k')
            plt.xlabel('Area (pixels²)')
            plt.ylabel('Count')
            plt.title('Floe Size Distribution (pixels²)')
            plt.tight_layout()
            plt.show()
        else:
            print("No components above min_area; skipping size distribution plot.")


def _pick_points(img_rgb, n_points=8, title="Click points (close window when done)"):
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"{title}\nPick {n_points} points in order; close the window when done.")
    pts = plt.ginput(n_points, timeout=0)
    plt.close()
    if len(pts) < 4:
        raise ValueError("Need at least 4 points for a homography.")
    return np.array(pts, dtype=np.float32)

def _warp_labels_nearest(labels_int, H, out_w, out_h):
    warped = cv.warpPerspective(labels_int.astype(np.int32), H, (out_w, out_h),
                                flags=cv.INTER_NEAREST,
                                borderMode=cv.BORDER_CONSTANT, borderValue=0)
    return warped

def _draw_contours_and_ids_on_visible(visible_rgb, labels_vis, centroids_vis):
    overlay_rgb = visible_rgb.copy()
    count = int(labels_vis.max())
    for lbl in range(1, count + 1):
        comp_mask = (labels_vis == lbl).astype(np.uint8) * 255
        if comp_mask.sum() == 0:
            continue
        contours, _ = cv.findContours(comp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(overlay_rgb, contours, -1, (255, 0, 0), 1, lineType=cv.LINE_AA)
    overlay_bgr = cv.cvtColor(overlay_rgb, cv.COLOR_RGB2BGR)
    Hh, Ww = overlay_bgr.shape[:2]
    font_scale = max(0.5, min(2.0, Hh / 1000.0))
    for idx, (x, y) in enumerate(centroids_vis, start=1):
        xi, yi = int(round(x)), int(round(y))
        cv.putText(overlay_bgr, str(idx), (xi, yi),
                   cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(overlay_bgr, str(idx), (xi, yi),
                   cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)
    return cv.cvtColor(overlay_bgr, cv.COLOR_BGR2RGB)

def register_and_overlay_visible(thermal_gray, thermal_labels, thermal_centroids,
                                 visible_rgb, n_points=8, ransac_thresh=3.0):
    th_rgb_for_click = cv.cvtColor(thermal_gray, cv.COLOR_GRAY2RGB)
    pts_th = _pick_points(th_rgb_for_click, n_points=n_points, title="THERMAL: pick points")
    pts_vis = _pick_points(visible_rgb, n_points=n_points, title="VISIBLE: pick SAME points in SAME order")

    H, inliers = cv.findHomography(pts_th, pts_vis, method=cv.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H is None:
        raise RuntimeError("Homography estimation failed. Try different / more points.")

    vis_h, vis_w = visible_rgb.shape[:2]
    warped_labels = _warp_labels_nearest(thermal_labels, H, vis_w, vis_h)

    if len(thermal_centroids) > 0:
        pts = np.array(thermal_centroids, dtype=np.float32).reshape(-1, 1, 2)
        pts_warp = cv.perspectiveTransform(pts, H).reshape(-1, 2)
    else:
        pts_warp = np.zeros((0, 2), dtype=np.float32)

    overlay_numbered = _draw_contours_and_ids_on_visible(visible_rgb, warped_labels, pts_warp)
    return overlay_numbered

def match_floes(labels1, centroids1, areas1,
                labels2, centroids2, areas2,
                max_centroid_dist=50,  # pixels
                min_iou=0.05):
    """
    Match floes between two frames using centroid distance + IoU.

    labels1, labels2: 2D int arrays, labels in [0..N]
    centroids1, centroids2: list of (cx, cy) for each label index 1..N
    areas1, areas2: list of areas (same ordering as centroids)
    Returns:
        matches: list of (idx1, idx2, dx, dy, dist, iou)
                 where idx1, idx2 are 0-based indices into centroids1/centroids2
        unmatched1: list of idx1 with no match
        unmatched2: list of idx2 with no match
    """
    H, W = labels1.shape
    N1 = len(centroids1)
    N2 = len(centroids2)

    unmatched2 = set(range(N2))
    matches = []
    unmatched1 = []

    for i in range(N1):
        cx1, cy1 = centroids1[i]
        lbl1 = i + 1  # label value in labels1

        best_j = None
        best_score = -np.inf
        best_iou = 0.0
        best_dist = np.inf

        for j in list(unmatched2):
            cx2, cy2 = centroids2[j]
            lbl2 = j + 1  # label value in labels2

            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = np.hypot(dx, dy)
            if dist > max_centroid_dist:
                continue  # too far to be plausible

            # IoU between the two floe masks
            mask1 = (labels1 == lbl1)
            mask2 = (labels2 == lbl2)
            inter = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            iou = inter / union if union > 0 else 0.0
            if iou < min_iou:
                continue

            # Score: prefer high IoU, small distance
            score = 5.0 * iou - 0.01 * dist

            if score > best_score:
                best_score = score
                best_j = j
                best_iou = iou
                best_dist = dist

        if best_j is not None:
            cx2, cy2 = centroids2[best_j]
            dx = cx2 - cx1
            dy = cy2 - cy1
            matches.append((i, best_j, dx, dy, best_dist, best_iou))
            unmatched2.remove(best_j)
        else:
            unmatched1.append(i)

    unmatched2 = list(unmatched2)
    return matches, unmatched1, unmatched2


def visualize_motion(img_gray, centroids1, matches,
                     save_path='floe_motion_vectors.png'):
    """
    Visualize floe motion as arrows from centroids in frame 1.

    img_gray: grayscale image from frame 1
    centroids1: list of (cx, cy) for frame 1
    matches: list of (idx1, idx2, dx, dy, dist, iou)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img_gray, cmap='gray')
    plt.title("Floe motion vectors (t0 → t1)")
    plt.axis('off')

    xs, ys, us, vs = [], [], [], []

    for idx1, idx2, dx, dy, dist, iou in matches:
        cx, cy = centroids1[idx1]
        xs.append(cx)
        ys.append(cy)
        us.append(dx)
        vs.append(dy)

    if len(xs) > 0:
        plt.quiver(xs, ys, us, vs,
                   angles='xy', scale_units='xy', scale=1,
                   color='red', width=0.002)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    img1_path = 'sea_ice_analysis/sea_ice_thermal.jpg'
    img2_path = 'sea_ice_analysis/sea_ice_thermal2.jpg'

    processor1= FloeSeparator(img1_path)
    processor2 = FloeSeparator(img2_path)

    # run preprocess to populate self.img_gray, self.img_rgb, self.mask
    img_gray, img_rgb, mask = processor1.preprocess(clahe_clip=1.5, clahe_grid=(8,8), thresh_val=70)
    img_gray2, img_rgb2, mask2 = processor2.preprocess(clahe_clip=1.5, clahe_grid=(8,8), thresh_val=70)

    # select one of the splitting methods:
    # labels_filtered, areas_px, centroids = processor1.split_simple(mask, min_area=30) # no erosion, just use binary mask
    # labels_filtered, areas_px, centroids = processor1.split_erode_dilate(mask, min_area=10) # 1px erosion/dilation
    labels_filtered, areas_px, centroids = processor1.split_erode_dilate_small(mask, min_area=30) # 1px erosion/dilation
    labels_filtered2, areas_px2, centroids2 = processor2.split_erode_dilate_small(mask2, min_area=30)

    # below code visualizes and saves outputs
    # processor1.visualize_and_save(img_gray, img_rgb, mask, labels_filtered, centroids, areas_px)

    # match floes between frame 1 and 2
    matches, unmatched1, unmatched2 = match_floes(
        labels_filtered,  centroids,  areas_px,
        labels_filtered2, centroids2, areas_px2,
        max_centroid_dist=40,   # tweak based on expected motion
        min_iou=0.05            # tweak based on overlap
    )

    print(f"Total floes t0: {len(centroids)}, t1: {len(centroids2)}")
    print(f"Matched floes: {len(matches)}")
    print(f"Unmatched in t0 (no match in t1): {unmatched1}")
    print(f"Unmatched in t1 (new / disappeared): {unmatched2}")

    for i, j, dx, dy, dist, iou in matches:
        print(f"floe {i+1} → floe {j+1}: "
              f"dx={dx:.2f}, dy={dy:.2f}, |d|={dist:.2f}, IoU={iou:.3f}")

    # visualize motion on the first frame
    visualize_motion(img_gray, centroids, matches,
                     save_path='floe_motion_vectors.png')


    # visible_path = 'sea_ice_analysis/sea_ice.jpg'
    # vis_bgr = cv.imread(visible_path)
    # if vis_bgr is None:
    #     raise FileNotFoundError(f"Could not read visible image at: {os.path.abspath(visible_path)}")
    # vis_rgb = cv.cvtColor(vis_bgr, cv.COLOR_BGR2RGB)

    # overlay_numbered_vis = register_and_overlay_visible(
    #     thermal_gray=img_gray,
    #     thermal_labels=labels_filtered,
    #     thermal_centroids=centroids,
    #     visible_rgb=vis_rgb,
    #     n_points=4,
    #     ransac_thresh=3.0
    # )

    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.title("Visible image")
    # plt.imshow(vis_rgb)
    # plt.axis('off')
    # plt.subplot(122)
    # plt.title("Thermal edges overlaid on visible")
    # plt.imshow(overlay_numbered_vis)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('visible_overlay_preview.png', dpi=150)
    # plt.show()
