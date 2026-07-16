"""Train Go2 walk policy: python -m train"""

import absl.logging

# Flax uses a local I/O shim when TensorFlow is absent. Safe for local
# checkpoints; only gs:// paths would need tensorflow.io.gfile.
absl.logging.set_verbosity(absl.logging.ERROR)

from train.main import main

if __name__ == '__main__':
    raise SystemExit(main())
