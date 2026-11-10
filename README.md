# harvest-cuda

CUDA-accelerated implementation of WORLD's Harvest F0 estimator, with a `pyworld.harvest`-compatible NumPy API.

Requires Python 3.10+ on Linux x86_64 (glibc 2.28+) or Windows x64,
and an NVIDIA GPU (compute capability 7.5+) with a CUDA 13-compatible driver.
Windows also requires the [Visual C++ Redistributable (x64)](https://aka.ms/vc14/vc_redist.x64.exe).

## Usage

```sh
pip install harvest-cuda
```

```python
import numpy as np
from harvest_cuda import harvest

x = np.ascontiguousarray(audio, dtype=np.float64)
f0, t = harvest(x, fs)
```

Use `harvest(x, fs, device=0)` to select a GPU; indices follow `CUDA_VISIBLE_DEVICES`.
The default is the first visible GPU.

`harvest()` leaves cached working memory for reuse.
Call `harvest_cuda.clear_cache()` to release it.

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [third-party notices](third_party/NOTICES.txt).
