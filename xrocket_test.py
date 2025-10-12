from xrocket import XRocket
import torch
import numpy as np


def print_tensor(x):
    s=np.array2string(np.array(x.numpy()), separator=' ', max_line_width=10**9)
    print(s.replace('\n\n', '\n'))


in_channels = 2
rocket = XRocket(in_channels=in_channels, max_kernel_span=10, combination_order=2, feature_cap=9*2, kernel_length=3, max_dilations=1)

print("XRocket dilations: ", rocket.dilations)
print("XRocket num_kernels: ", rocket.num_kernels)
print("XRocket num_combinations: ", rocket.num_combinations)
print("XRocket num_thresholds: ", rocket.num_thresholds)
print("XRocket num_dilations: ", rocket.num_dilations)
print("XRocket num_features: ", rocket.num_features)

# fit rocket with random data
rocket.fit(torch.zeros(1, in_channels, 10))

num_samples = 2#len(rocket.blocks[0].conv.weight)
samples = torch.zeros(num_samples, in_channels, 10)
i=0
for block in rocket.blocks:
    for weight in block.conv.weight:
        samples[i,0,3:weight.shape[1]+3] = weight[0]
        samples[i,1,3:weight.shape[1]+3] = 0
        i+=1
        if i>=num_samples:
            break

for block in rocket.blocks:
    print("Block conv weight:"); print_tensor(block.conv.weight.data)
    print("Block conv patterns:"); print(block.conv.patterns)
    print("Mix weight: "); print_tensor(block.mix.weight)
    print("Mix combinations: "); print(block.mix.combinations)
    print("Thresholds Bias: "); print_tensor(block.thresholds.bias)
    print("Thresholds Thresholds: "); print(block.thresholds.thresholds)
    print("Samples: ", samples.shape); print_tensor(samples)
    x = block.conv(samples)
    print("Conv on samples: ", x.shape); print_tensor(x)
    x = block.mix(x)
    print("Mix on samples: ", x.shape); print_tensor(x)
    x = block.thresholds(x)
    x = torch.flatten(x, start_dim=1)
    print("Flattened Thresholds on samples: ", x.shape); print_tensor(x)
    print("Flattened Thresholds on samples: "); print_tensor(x)

print("Num features: ", rocket.num_features)
print("Feature names: ");
print()
for i,name in enumerate(rocket.blocks[0].feature_names):
    print(i, name, x[:,i])
