import pickle, numpy as np, os, glob

data_root = 'C:/Users/DIAT/ashish/hpe/steins_shrinkage/MotionAGFormer/data/motion3d/H36M-243/test'
files = sorted(glob.glob(os.path.join(data_root, '*.pkl')))
print(f'Total test files: {len(files)}')

with open(files[0], 'rb') as f:
    d = pickle.load(f)

print(f'Keys: {list(d.keys())}')
label = d['data_label']
inp = d['data_input']
print(f'data_label shape: {label.shape}, dtype: {label.dtype}')
print(f'data_input shape: {inp.shape}, dtype: {inp.dtype}')
print(f'data_label range: min={label.min():.6f}, max={label.max():.6f}, mean={label.mean():.6f}, std={label.std():.6f}')
print(f'data_input range: min={inp.min():.6f}, max={inp.max():.6f}')

label_rr = label - label[:, 0:1, :]
print(f'Root-relative label range: min={label_rr.min():.6f}, max={label_rr.max():.6f}, std={label_rr.std():.6f}')

# Bone lengths in normalized coords
hip_to_knee = np.linalg.norm(label[:, 1, :] - label[:, 0, :], axis=-1)
print(f'Hip-knee bone length (norm): mean={hip_to_knee.mean():.6f}')

# A real human hip-to-knee is ~400mm. So factor = 400 / mean_bone_length
real_hip_knee_mm = 400.0
factor = real_hip_knee_mm / hip_to_knee.mean()
print(f'Estimated denorm factor (from hip-knee): {factor:.1f}')
