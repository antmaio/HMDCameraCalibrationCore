import sys
import os
import cv2
import json
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),  '..')))

import config

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)

def unity_to_cv(pos, rot_q):
    """
    Map Unity's tracking coordinate space (LH: x-rt, y-up, z-fwd) 
    to OpenCV coordinate space (RH: x-rt, y-dn, z-fwd).
    """
    pos_cv = np.array([pos[0], -pos[1], pos[2]], dtype=np.float32)
    R_u = quaternion_to_rotation_matrix(rot_q)
    M = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    R_cv = M @ R_u @ M
    return pos_cv, R_cv

def find_chessboard_corners_single(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners_refined.reshape(-1, 2)
    return None

def project_point(P, pt_3d):
    pt_4d = np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0], dtype=np.float32)
    uvw = P @ pt_4d
    z = uvw[2]
    if z <= 0.0:
        return None
    return (int(uvw[0]/z), int(uvw[1]/z))

def main():
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snapshot', 'ForHMDCalibrationOnlySnapshot'))
    if not os.path.exists(folder_path):
        print(f"[ERROR] Snapshot folder not found: {folder_path}\nMake sure you have placed your snapshots there.")
        sys.exit(1)
        
    json_files = sorted(glob.glob(os.path.join(folder_path, "Snapshot_*.json")))
    if not json_files:
        print(f"[ERROR] No JSON files found in {folder_path}.")
        sys.exit(1)
        
    w, h = config.CALIBRATION_BOARD_SIZE
    sq_size = config.SQUARE_SIZE
    
    # Generate 3D points in board local coordinate system (z=0)
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * sq_size

    global_target_spaces = set(["trackingSpace", "guardianSpace", "anchorSpace"])

    # Load all valid snapshots
    snapshots = []
    for jf in json_files:
        pf = jf.replace(".json", ".png")
        if not os.path.exists(pf):
            # Fallback to jpg if png doesn't exist
            pf = jf.replace(".json", ".jpg")
            if not os.path.exists(pf):
                print(f"[WARNING] Missing image for {jf}")
                continue
            
        with open(jf, "r") as f:
            data = json.load(f)
            
        img = cv2.imread(pf)
        if img is None:
            print(f"[WARNING] Could not read image {pf}")
            continue
            
        corners = find_chessboard_corners_single(img, (w, h))
        if corners is None:
            print(f"[WARNING] Chessboard not found in {os.path.basename(pf)}")
            continue
            
        space_transforms = {}
        target_spaces_current = ["trackingSpace", "guardianSpace", "anchorSpace"]
        for sp in target_spaces_current:
            sp_data = data.get(sp, {})
            # Handle root-level case for tracking space
            if not sp_data and "left_camera_position" in data and sp == "trackingSpace":
                sp_data = data
                
            if "left_camera_position" in sp_data and "left_camera_rotation" in sp_data:
                pos = sp_data["left_camera_position"]
                rot = sp_data["left_camera_rotation"]
                pos_u = np.array([pos["x"], pos["y"], pos["z"]])
                rot_u = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])
                pos_cv, R_cv = unity_to_cv(pos_u, rot_u)
                space_transforms[sp] = {"R_c2s": R_cv, "t_c2s": pos_cv}
            else:
                space_transforms[sp] = None

        if data.get("hasSceneSpaces", False) and "sceneSpaces" in data:
            for scene_space in data["sceneSpaces"]:
                anchor_name = f"sceneSpace_{scene_space['anchorName']}"
                global_target_spaces.add(anchor_name)
                sp_data = scene_space.get("poseData", {})
                if "left_camera_position" in sp_data and "left_camera_rotation" in sp_data:
                    pos = sp_data["left_camera_position"]
                    rot = sp_data["left_camera_rotation"]
                    pos_u = np.array([pos["x"], pos["y"], pos["z"]])
                    rot_u = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])
                    pos_cv, R_cv = unity_to_cv(pos_u, rot_u)
                    space_transforms[anchor_name] = {"R_c2s": R_cv, "t_c2s": pos_cv}

        # Camera intrinsics
        fx = data["focalLength"]["x"]
        fy = data["focalLength"]["y"]
        cx = data["principalPoint"]["x"]
        cy = data["principalPoint"]["y"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, None)
        if ret:
            R_b2c, _ = cv2.Rodrigues(rvec)
            t_b2c = tvec.reshape(3)
        else:
            print(f"[WARNING] solvePnP failed for {os.path.basename(pf)}")
            continue
        
        snapshots.append({
            "json_path": jf,
            "png_path": pf,
            "image": img,
            "corners_2d": corners,
            "K": K,
            "space_transforms": space_transforms,
            "R_b2c": R_b2c,
            "t_b2c": t_b2c
        })

    if not snapshots:
        print("[ERROR] No valid snapshots with chessboard found.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(snapshots)} valid snapshots.")

    target_spaces = sorted(list(global_target_spaces))
    results_by_space = {sp: [] for sp in target_spaces}
    results_3d_by_space = {sp: [] for sp in target_spaces}

    # Round robin
    for i, ref_snap in enumerate(snapshots):
        print(f"\n[INFO] Reference Snapshot: {os.path.basename(ref_snap['json_path'])}")
        
        R_b2c = ref_snap["R_b2c"]
        t_b2c = ref_snap["t_b2c"]
        
        errors_for_ref = {sp: [] for sp in target_spaces}
        errors_3d_for_ref = {sp: [] for sp in target_spaces}
        
        for space in target_spaces:
            ref_tx = ref_snap["space_transforms"].get(space)
            if not ref_tx: continue
            
            # Transform board corners from Local Board space -> Ref Camera Space -> Target Space
            corners_3d_space = []
            for pt in objp:
                p_cam = R_b2c @ pt + t_b2c
                p_space = ref_tx["R_c2s"] @ p_cam + ref_tx["t_c2s"]
                corners_3d_space.append(p_space)
            
            corners_3d_space = np.array(corners_3d_space)

            # Test against all other snapshots
            for j, test_snap in enumerate(snapshots):
                if i == j: continue
                
                test_tx = test_snap["space_transforms"].get(space)
                if not test_tx: continue
                
                # 3D points in target space according to test snapshot
                test_corners_3d_space = []
                for pt in objp:
                    p_cam = test_snap["R_b2c"] @ pt + test_snap["t_b2c"]
                    p_space = test_tx["R_c2s"] @ p_cam + test_tx["t_c2s"]
                    test_corners_3d_space.append(p_space)
                test_corners_3d_space = np.array(test_corners_3d_space)
                
                # Compute 3D error in cm (assuming Unity units are meters)
                dist_3d = np.linalg.norm(corners_3d_space - test_corners_3d_space, axis=1) * 100
                errors_3d_for_ref[space].append(np.mean(dist_3d))

                R_s2c = test_tx["R_c2s"].T
                t_s2c = -R_s2c @ test_tx["t_c2s"]
                
                P = test_snap["K"] @ np.hstack([R_s2c, t_s2c.reshape(3,1)])
                gt_corners = test_snap["corners_2d"]
                
                errs = []
                display_img = test_snap["image"].copy()
                
                for pt_idx, pt_3d in enumerate(corners_3d_space):
                    proj_2d = project_point(P, pt_3d)
                    if proj_2d:
                        orig_2d = gt_corners[pt_idx]
                        dx = proj_2d[0] - orig_2d[0]
                        dy = proj_2d[1] - orig_2d[1]
                        dist = np.sqrt(dx**2 + dy**2)
                        errs.append(dist)
                        
                        # Yellow line, red reprojection, green ground truth
                        cv2.circle(display_img, (int(orig_2d[0]), int(orig_2d[1])), 4, (0, 255, 0), 2)
                        cv2.circle(display_img, proj_2d, 4, (0, 0, 255), 2)
                        cv2.line(display_img, (int(orig_2d[0]), int(orig_2d[1])), proj_2d, (0, 255, 255), 1)

                if errs:
                    errors_for_ref[space].append(np.mean(errs))
                    
                    # Save visual debug image
                    ref_name = os.path.basename(ref_snap['json_path']).replace(".json", "")
                    test_name = os.path.basename(test_snap['json_path']).replace(".json", "")
                    
                    out_folder = os.path.join(folder_path, "PnP_Evaluation")
                    os.makedirs(out_folder, exist_ok=True)
                    out_path = os.path.join(out_folder, f"Ref_{ref_name}__Test_{test_name}_{space}.png")
                    cv2.imwrite(out_path, display_img)
                    
        for space in target_spaces:
            if errors_for_ref[space]:
                overall_mean = np.mean(errors_for_ref[space])
                overall_mean_3d = np.mean(errors_3d_for_ref[space])
                print(f"  [{space}] => Mean reprojection error: {overall_mean:.2f} px | Mean 3D error: {overall_mean_3d:.2f} cm")
                results_by_space[space].append({
                    "reference": os.path.basename(ref_snap["json_path"]),
                    "mean_error": overall_mean
                })
                results_3d_by_space[space].append({
                    "reference": os.path.basename(ref_snap["json_path"]),
                    "mean_error_3d": overall_mean_3d
                })

    print("\n[SUMMARY]")
    for space in target_spaces:
        if results_by_space[space]:
            total_mean = np.mean([r["mean_error"] for r in results_by_space[space]])
            total_mean_3d = np.mean([r["mean_error_3d"] for r in results_3d_by_space[space]])
            print(f"  Overall {space} Cross-Reprojection Mean Error: {total_mean:.2f} px | 3D Error: {total_mean_3d:.2f} cm")
            
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    all_refs = []
    for space in target_spaces:
        for r in results_by_space[space]:
            if r["reference"] not in all_refs:
                all_refs.append(r["reference"])
                
    x = np.arange(len(all_refs))
    width = 0.8 / len(target_spaces)
    
    def get_means(sp, results_dict, key):
        means = []
        for ref in all_refs:
            val = next((r[key] for r in results_dict[sp] if r["reference"] == ref), 0)
            means.append(val)
        return means

    colors_pool = ['skyblue', 'lightgreen', 'salmon', 'plum', 'khaki', 'coral', 'tan', 'gold', 'cyan', 'lavender']
    for idx, space in enumerate(target_spaces):
        means_2d = get_means(space, results_by_space, "mean_error")
        means_3d = get_means(space, results_3d_by_space, "mean_error_3d")
        color = colors_pool[idx % len(colors_pool)]
        offset = (idx - len(target_spaces) / 2.0 + 0.5) * width
        
        ax1.bar(x + offset, means_2d, width, label=space, color=color, edgecolor='black')
        ax2.bar(x + offset, means_3d, width, label=space, color=color, edgecolor='black')
        
    clean_refs = [ref.replace("Snapshot_", "").replace(".json", "") for ref in all_refs]
    
    ax1.set_ylabel("Mean Reprojection Error (pixels)")
    ax1.set_title("PnP Cross-Reprojection Error Comparison by Coordinate Space")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_refs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.set_ylabel("Mean 3D Error (cm)")
    ax2.set_xlabel("Reference Snapshot")
    ax2.set_title("PnP 3D Position Error Comparison by Coordinate Space (in cm)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(clean_refs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plot_path = os.path.join(folder_path, "PnP_Evaluation", "pnp_cross_reprojection_report.png")
    plt.savefig(plot_path)
    print(f"[INFO] Plot saved to {plot_path}")

if __name__ == "__main__":
    main()